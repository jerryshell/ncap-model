import os
import time

import tensorflow.keras as keras
from flask import Flask
from flask_restful import reqparse, Api, Resource

from data_helper import DataHelper

# 超参数
feature1_number = 60  # 句子分成多少个词语，多余截断，不够补 0
feature2_number = 300  # 每个词语的向量

print('model loading...')
# 加载模型
model = keras.models.load_model("save.h5")

# 加载数据
print('vector loading...')
data_helper = DataHelper(feature1_number, feature2_number)

# flask 初始化
app = Flask(__name__)
api = Api(app)

# 请求参数处理器
parser = reqparse.RequestParser()
parser.add_argument('sentence', type=str, required=True, help='need sentence data')
parser.add_argument('token', type=str, required=True, help='need token data')


# 加载 token 列表
def load_token_list():
    with open('./token_list', 'r') as f:
        return [token.strip() for token in f.readlines()]


class Index(Resource):
    def get(self):
        return {'Auth': 'Jerry', 'GitHub': 'https://github.com/jerryshell'}

    def post(self):
        # 解析请求参数
        args = parser.parse_args()
        sentence = args['sentence']
        token = args['token']
        # 加载 token 列表
        token_list = load_token_list()
        if token not in token_list:
            return {'ok': False, 'message': 'token error'}
        # 调用模型获得结果
        test_data = data_helper.sentence2test_data(sentence)
        result = model.predict(test_data)
        a = result[0][0] * 100
        b = result[0][1] * 100
        c = result[0][2] * 100
        d = result[0][3] * 100
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 保存历史记录
        history = '%s %s %s %s %s %s %s' % (token, time_str, sentence, a, b, c, d)
        os.system('echo "%s" >> history' % history)
        print(history)
        # 返回响应
        return {'ok': True, 'a': a, 'b': b, 'c': c, 'd': d}


# 绑定路由
api.add_resource(Index, '/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
