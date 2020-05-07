import re

import jieba
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from data_loader import DataLoader


class DataHelper:
    def __init__(self, feature1_number, feature2_number):
        self.feature1_number = feature1_number
        self.feature2_number = feature2_number

        # 加载字典
        self.gensim_model = KeyedVectors.load_word2vec_format('./data/sgns.wiki.bigram-char')

        # 加载原始数据
        data_loader = DataLoader()

        # 原始训练数据生成器
        self.raw_train_data_generator = data_loader.train_data_generator()
        # 原始验证数据生成器
        self.raw_validation_data_generator = data_loader.validation_data_generator()
        # 原始测试数据生成器
        self.raw_test_data_generator = data_loader.test_data_generator()

        # 训练数据大小
        self.train_data_count = data_loader.train_data_count
        # 验证数据大小
        self.validation_data_count = data_loader.validation_data_count
        # 测试数据大小
        self.test_data_count = data_loader.test_data_count

    # 把一个单独的词语转换成向量，如果不存在则返回 0
    def word2vec(self, word):
        if word in self.gensim_model:
            return self.gensim_model.get_vector(word)
        return np.zeros(shape=(self.feature2_number,), dtype=np.float32)

    # 把词语列表转换成向量，版本 2，从最后一个词开始填充 vector
    def word_list2vector_tail(self, word_list):
        word_list_len = len(word_list)
        vector = np.zeros(shape=(self.feature1_number, self.feature2_number))
        if word_list_len <= self.feature1_number:
            for i in range(self.feature1_number):
                if i < word_list_len:
                    word = word_list[word_list_len - 1 - i]
                    vec = self.word2vec(word)
                    vector[self.feature1_number - 1 - i] = vec
        else:
            for i in range(self.feature1_number):
                if i < word_list_len:
                    word = word_list[i]
                    vec = self.word2vec(word)
                    vector[i] = vec
        return vector

    # 把词语列表转换成向量
    def word_list2vector(self, word_list):
        word_list_len = len(word_list)
        vector = np.zeros(shape=(self.feature1_number, self.feature2_number))
        for i in range(self.feature1_number):
            if i < word_list_len:
                word = word_list[i]
                vec = self.word2vec(word)
                vector[i] = vec
            else:
                break
        return vector

    # data_generator 中读取 batch_size 个数据，并转换成向量返回
    def get_batch_vector_and_label(self, data_generator: iter, batch_size):
        # 初始化返回结果
        batch_label = np.zeros(shape=(batch_size,))
        batch_vector = np.zeros(shape=(batch_size, self.feature1_number, self.feature2_number))
        # 根据 batch_size 填充返回结果
        for batch_index in range(batch_size):
            # 从 data_generator 中读取下一个数据
            sentence, label = next(data_generator)
            # 分词，获得词语列表
            word_list = sentence.split(' ')
            # 把词语列表转换成向量
            vector = self.word_list2vector(word_list)
            # 填充返回结果
            batch_label[batch_index] = label
            batch_vector[batch_index] = vector
        return batch_vector, batch_label

    # 训练数据生成器
    def train_data_generator(self, batch_size):
        while True:
            yield self.get_batch_vector_and_label(self.raw_train_data_generator, batch_size)

    # 验证数据生成器
    def validation_data_generator(self, batch_size):
        while True:
            yield self.get_batch_vector_and_label(self.raw_validation_data_generator, batch_size)

    # 测试数据生成器
    def test_data_generator(self, batch_size):
        while True:
            yield self.get_batch_vector_and_label(self.raw_test_data_generator, batch_size)

    # 将一句话转成向量
    def sentence2vector(self, sentence):
        # 去特殊字符
        sentence = re.sub(r'\W+', ' ', sentence).replace('_', ' ')
        # 分词
        word_list = jieba.lcut(sentence)
        # 去空格
        word_list = [item for item in word_list if item != ' ']
        # 把词语列表转换成向量
        vector = self.word_list2vector(word_list)
        return vector

    # 将一句话转成 batch_size 为 1 的测试数据
    def sentence2batch_vector(self, sentence):
        test_data = np.zeros(shape=(1, self.feature1_number, self.feature2_number))
        test_data[0] = self.sentence2vector(sentence)
        return test_data
