# TextCNN 文本情感分类

* 数据：https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/simplifyweibo_4_moods/intro.ipynb
* 词向量：https://github.com/Embedding/Chinese-Word-Vectors

## 依赖

### 升级 pip 并设置镜像

```bash
pip3 install --upgrade pip -i https://pypi.douban.com/simple
pip3 config set global.index-url https://pypi.douban.com/simple
```

### 安装依赖

```bash
pip3 install --upgrade tensorflow keras pandas numpy jieba gensim fastapi uvicorn
```

## Docker

```bash
mkdir -p /docker-data/tf/notebooks
```

```bash
docker run -it --rm \
  -v $PWD:/tmp \
  -w /tmp \
  tensorflow/tensorflow:latest-py3-jupyter \
  pip3 install --upgrade pip -i https://pypi.douban.com/simple && \
  pip3 config set global.index-url https://pypi.douban.com/simple && \
  pip3 install tensorflow keras pandas numpy jieba gensim fastapi uvicorn && \
  python3 model_train.py 64 100 1>log 2>&1
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
