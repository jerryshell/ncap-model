# 新闻评论分析平台情感分析模块

* 训练数据：https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/simplifyweibo_4_moods/intro.ipynb
* 词向量：https://github.com/Embedding/Chinese-Word-Vectors

## 相关项目

* [新闻评论分析平台 Web 端](https://github.com/jerryshell/ncap-admin)
* [新闻评论分析平台数据大屏](https://github.com/jerryshell/ncap-datav)
* [新闻评论分析平台服务端](https://github.com/jerryshell/ncap-server)
* 新闻评论分析平台情感分析模块（当前项目）
* [新闻评论分析平台数据采集模块](https://github.com/jerryshell/ncap-spider)

## 环境配置

### 升级 pip 并设置镜像

```bash
pip3 install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple
```

#### aliyun

```bash
pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple
```

#### douban

```bash
pip3 config set global.index-url https://pypi.douban.com/simple
```

### 安装依赖

```bash
pip3 install -U tensorflow keras pandas numpy jieba fastapi uvicorn
```

## Docker

### 模型训练

```bash
docker run -d --rm --name tf \
  -v $PWD:/data \
  -w /data \
  tensorflow/tensorflow:latest \
  bash -c ' \
    pip3 install -U pip -i https://mirrors.aliyun.com/pypi/simple && \
    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip3 install -U tensorflow keras pandas numpy jieba && \
    python3 model_train.py 64 100 false 1>log 2>&1 \
  '
```

### tensorboard

```bash
docker run -d --rm --name tf-board \
  -p 6006:6006 \
  -v $PWD:/data \
  -w /data \
  tensorflow/tensorflow:latest \
  tensorboard --logdir logs/fit --host 0.0.0.0 --port 6006
```

### 开启 HTTP API

```bash
docker run -d --rm --name tf-api \
  -p 8000:8000 \
  -v $PWD:/data \
  -w /data \
  tensorflow/tensorflow:latest \
  bash -c ' \
    pip3 install -U pip -i https://mirrors.aliyun.com/pypi/simple && \
    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip3 install -U tensorflow keras pandas numpy jieba fastapi uvicorn && \
    python3 api.py text_cnn.2.79.h5 1>api_log 2>&1 \
  '
```

## 参考

* https://tf.wiki
* https://www.tensorflow.org/guide/keras/save_and_serialize#part_ii_saving_and_loading_of_subclassed_models
* https://zhuanlan.zhihu.com/p/25630700
* https://blog.csdn.net/asialee_bird/article/details/88813385
* https://trickygo.github.io/Dive-into-DL-TensorFlow2.0
* https://www.zybuluo.com/Dounm/note/591752
* https://zhuanlan.zhihu.com/p/54397748
* https://www.cnblogs.com/jiangxinyang/p/10241243.html
