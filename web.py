import os
import time

import numpy as np
import tensorflow.keras as keras
from flask import Flask, request
from flask_restful import reqparse, Api, Resource

from data_helper import DataHelper

# 网站公告
notice = ''

# 训练状态
train_status = {
    'real_time_tuning': True,  # 实时调教
    'color': 'green',  # 公告字体颜色
    'message': '调教参数实时更新',  # 公告信息
}

# 超参数
feature1_number = 60  # 句子分成多少个词语，多余截断，不够补 0
feature2_number = 300  # 每个词语的向量

# 加载模型
print('model loading...')
model_file_name = 'save_text_cnn'
model = keras.models.load_model(model_file_name + '.h5')

# 加载数据
print('vector loading...')
data_helper = DataHelper(feature1_number, feature2_number)

# flask 初始化
app = Flask(__name__)
api = Api(app)


@app.after_request
def after_request(response):
    # 允许跨域
    response.headers.add('Access-Control-Allow-Origin', '*')
    if request.method == 'OPTIONS':
        response.headers['Access-Control-Allow-Methods'] = 'POST, DELETE, PUT, GET'
        headers = request.headers.get('Access-Control-Request-Headers')
        if headers:
            response.headers['Access-Control-Allow-Headers'] = headers
    return response


# 请求参数处理器
client_parser = reqparse.RequestParser()
client_parser.add_argument('sentence', type=str, required=True, help='need sentence data')
client_parser.add_argument('token', type=str, required=True, help='need token data')
client_parser.add_argument('trainFlag', type=bool, required=False)
client_parser.add_argument('trainLabel', type=int, required=False)

admin_parser = reqparse.RequestParser()
admin_parser.add_argument('token', type=str, required=True, help='need token data')
admin_parser.add_argument('key', type=str, required=True, help='need key data')
admin_parser.add_argument('value', type=str, required=True, help='need value data')

snapshot_parser = reqparse.RequestParser()
snapshot_parser.add_argument('token', type=str, required=True, help='need token data')

mdreload_parser = reqparse.RequestParser()
mdreload_parser.add_argument('token', type=str, required=True, help='need token data')
mdreload_parser.add_argument('name', type=str, required=True, help='need name data')


# 加载 token 列表
def load_token_list():
    with open('./token_list', 'r') as f:
        return [token.strip() for token in f.readlines()]


class Index(Resource):
    def get(self):
        return {'Auth': 'Jerry', 'GitHub': 'https://github.com/jerryshell'}

    def post(self):
        # 解析请求参数
        args = client_parser.parse_args()
        sentence = args['sentence']
        token = args['token']
        train_flag = args['trainFlag']
        train_label = args['trainLabel']

        # 加载 token 列表
        token_list = load_token_list()
        if token not in token_list:
            return {'ok': False, 'message': '请输入正确的 Token'}

        # 调教模式保存数据
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if train_flag:
            extra_data = '%s,%s,%s,%s' % (train_label, sentence, time_str, token)
            os.system('echo "%s" >> extra_data' % extra_data)
            # 实时调教
            if train_status['real_time_tuning']:
                train_label_np = np.zeros(shape=(1, 1))
                train_label_np.put(0, train_label)
                test_data = data_helper.sentence2test_data(sentence)
                model.fit(test_data, train_label_np)

        # 调用模型获得结果
        test_data = data_helper.sentence2test_data(sentence)
        result = model.predict(test_data)
        a = result[0][0] * 100
        b = result[0][1] * 100
        c = result[0][2] * 100
        d = result[0][3] * 100

        # 保存历史记录
        history = '%s %s %s %s %s %s %s' % (token, time_str, sentence, a, b, c, d)
        os.system('echo "%s" >> history' % history)
        print(history)

        # 返回响应
        return {'ok': True, 'a': a, 'b': b, 'c': c, 'd': d}


class Admin(Resource):
    def post(self):
        global train_status
        # 解析请求参数
        args = admin_parser.parse_args()
        token = args['token']
        key = args['key']
        value = args['value']
        print(args)

        if token != 'Super@dmin':
            return

        if key == 'set trainStatus.real_time_tuning':
            train_status['real_time_tuning'] = (value == 'open')
            return

        if key == 'set trainStatus.color':
            train_status['color'] = value
            return

        if key == 'set trainStatus.message':
            train_status['message'] = value
            return

        if key == 'set notice':
            global notice
            notice = value
            return


class Info(Resource):
    def get(self):
        return {
            'notice': notice,
            'trainStatus': train_status
        }


class Snapshot(Resource):
    def post(self):
        args = snapshot_parser.parse_args()
        token = args['token']
        if token != 'Super@dmin':
            return
        time_str = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime())
        model.save(model_file_name + '.' + time_str + '.h5')


class MDReload(Resource):
    def post(self):
        args = mdreload_parser.parse_args()
        token = args['token']
        name = args['name']
        if token != 'Super@dmin':
            return
        new_model = keras.models.load_model(name + '.h5')
        global model
        model = new_model


# 绑定路由
api.add_resource(Index, '/')
api.add_resource(Admin, '/zero')
api.add_resource(Info, '/info')
api.add_resource(Snapshot, '/snapshot')
api.add_resource(MDReload, '/mdreload')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8848)
