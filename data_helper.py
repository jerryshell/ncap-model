import re

import jieba
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


class DataHelper:
    def __init__(self, feature1_number, feature2_number):
        self.feature1_number = feature1_number
        self.feature2_number = feature2_number
        self.gensim_model = KeyedVectors.load_word2vec_format('sgns.wiki.bigram-char')

    def word2vec(self, word):
        if word in self.gensim_model:
            return self.gensim_model.get_vector(word)
        return np.zeros(shape=(self.feature2_number,), dtype=np.float32)

    def get_batch_label_and_vector(self, data_loader, batch_size):
        batch_label = np.zeros(shape=(batch_size,))
        batch_data = np.zeros(shape=(batch_size, self.feature1_number, self.feature2_number))
        for batch_index in range(batch_size):
            label, sentence = data_loader.next()
            # print(label, sentence)
            word_list = sentence.split(' ')
            word_list_len = len(word_list)
            # print(word_list)
            # print(word_list_len)
            # 从最后一个词开始填充 vector
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
            batch_label[batch_index] = label
            batch_data[batch_index] = vector
            # print(vector)
        return batch_label, batch_data

    # 将一句话转成向量
    def sentence2vector(self, sentence):
        # 去特殊字符
        sentence = re.sub(r'\W+', ' ', sentence).replace('_', ' ')
        # 分词
        word_list = jieba.lcut(sentence)
        # 去空格
        word_list = [item for item in word_list if item != ' ']
        word_list_len = len(word_list)
        print(word_list)
        # print(word_list_len)
        # 从最后一个词开始填充 vector
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
        # print(vector)
        return vector

    # 将一句话转成测试数据
    def sentence2test_data(self, sentence):
        test_data = np.zeros(shape=(1, self.feature1_number, self.feature2_number))
        test_data[0] = self.sentence2vector(sentence)
        return test_data


if __name__ == '__main__':
    from data_loader import DataLoader
    import sys

    feature1_number = 60
    feature2_number = 300
    data_loader = DataLoader()
    data_helper = DataHelper(feature1_number, feature2_number)
    np.set_printoptions(threshold=sys.maxsize)  # 强制打印 numpy 的整个数组
    print(data_helper.get_batch_label_and_vector(data_loader, 100))
