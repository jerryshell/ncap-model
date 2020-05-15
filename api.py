import gc
import os
import sys
import time

import numpy as np
import tensorflow.keras as keras
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_helper import DataHelper
from sysinfo.loadavg import loadavg
from sysinfo.meminfo import meminfo

if len(sys.argv) != 2:
    print('python3 api.py [model_filename]')
    exit(0)

print(sys.argv)

# 网站公告
notice = ''

# 训练状态
train_status = {
    # 实时调参
    'real_time_tuning': True,
    # 公告字体颜色
    'color': 'green',
    # 公告信息
    'message': '实时调参开启',
}

# 加载模型
print('model loading...')
model_filename = sys.argv[1]
model = keras.models.load_model(model_filename)
model.summary()

# 加载数据
print('vector loading...')
data_helper = DataHelper()

# 实例化 FastAPI
app = FastAPI()

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    # allow_origins=['http://localhost:8080'],
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# uniapp form
class UniappForm(BaseModel):
    sentence: str
    token: str
    train_flag: bool = Field(default=False, alias='trainFlag')
    train_label: int = Field(default=False, alias='trainLabel')


# admin form
class AdminForm(BaseModel):
    token: str
    key: str
    value: str


# admin form
class SnapshotForm(BaseModel):
    token: str


# model reload form
class ModelReloadForm(BaseModel):
    token: str
    model_filename: str


# 加载 token 列表
def load_token_list():
    with open('api_token_list', 'r') as f:
        return [token.strip() for token in f.readlines()]


# about me 接口
@app.get('/')
def index():
    return {'Auth': 'Jerry', 'GitHub': 'https://github.com/jerryshell'}


# uniapp 接口
@app.post('/')
def main(form: UniappForm):
    print(form)

    # 解析请求参数
    sentence_list = form.sentence.split('\n')
    token = form.token
    train_flag = form.train_flag
    train_label = form.train_label

    # 加载 token 列表
    token_list = load_token_list()
    if token not in token_list:
        return {'ok': False, 'message': '请输入正确的 Token'}

    # 多行使用时强制停用调参模式
    sentence_list_len = len(sentence_list)
    if sentence_list_len > 1:
        train_flag = False

    # 概率初始化
    result_p = 0.0
    result_n = 0.0

    # 迭代 sentence_list 使用模型得到概率
    for sentence in sentence_list:
        # 调参模式保存数据
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if train_flag:
            extra_data = '%s,%s,%s,%s' % (train_label, sentence, time_str, token)
            os.system('echo "%s" >> extra_data' % extra_data)
            # 实时调参
            if train_status['real_time_tuning']:
                train_label_np = np.zeros(shape=(1, 1))
                train_label_np.put(0, train_label)
                test_data = data_helper.sentence2idx_list(sentence)
                model.fit(test_data, train_label_np)

        # 调用模型获得结果
        test_data = data_helper.sentence2idx_list(sentence)
        print('sentence', sentence)
        print('sentence2idx_list', test_data)
        result = model.predict(test_data)
        p = result[0][0] * 100
        n = result[0][1] * 100

        # 保存历史记录
        history = '%s %s %s %s %s' % (token, time_str, sentence, p, n)
        os.system('echo "%s" >> history' % history)
        print(history)

        # 统计概率总和
        result_p += p
        result_n += n

    # 计算概率
    result_p /= sentence_list_len
    result_n /= sentence_list_len

    # 返回响应
    return {'ok': True, 'p': result_p, 'n': result_n}


# 管理接口
@app.post('/zero')
def admin(form: AdminForm):
    print(form)

    global train_status
    # 解析请求参数
    token = form.token
    key = form.key
    value = form.value

    if token != 'Super@dmin':
        return {'ok': False}

    if key == 'set trainStatus.realTimeTuning':
        train_status['real_time_tuning'] = (value == 'open')
        return {
            'ok': True
        }

    if key == 'set trainStatus.color':
        train_status['color'] = value
        return {
            'ok': True
        }

    if key == 'set trainStatus.message':
        train_status['message'] = value
        return {
            'ok': True
        }

    if key == 'set notice':
        global notice
        notice = value
        return {
            'ok': True
        }


# 服务器信息接口
@app.get('/info')
def info():
    model.summary()
    return {
        'ok': True,
        'loadavg': loadavg(),
        'meminfo': meminfo(),
        'notice': notice,
        'train_status': train_status,
        'model_filename': model_filename
    }


# 获取所有 model_filename
@app.get('/list/model_filename')
def list_model_filename():
    return os.popen('ls *.h5').read().strip().split('\n')


# 快照接口
@app.post('/snapshot')
def snapshot(form: SnapshotForm):
    print(form)

    if form.token != 'Super@dmin':
        return {
            'ok': False
        }

    time_str = time.strftime("%Y-%m-%d.%H.%M.%S", time.localtime())
    model.save(time_str + '.' + model_filename)
    return {
        'ok': True
    }


# 重载模型接口
@app.post('/modelReload')
def model_reload(form: ModelReloadForm):
    print(form)

    if form.token != 'Super@dmin':
        return {
            'ok': False
        }

    global model_filename
    model_filename = form.model_filename

    try:
        global model
        del model
        gc.collect()
        model = keras.models.load_model(model_filename)
        gc.collect()
        model.summary()
    except Exception as e:
        print(e)
        return {
            'ok': False,
            'message': 'keras.models.load_model(model_filename) fail'
        }

    return {
        'ok': True
    }


# 使用 uvicorn 运行 fastapi
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
