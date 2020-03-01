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
            label, data = data_loader.next()
            # print(label, data)
            word_list = data.split(' ')
            # print(word_list)
            # print(len(word_list))
            vector = np.zeros(shape=(self.feature1_number, self.feature2_number))
            for i in range(self.feature1_number):
                if i < len(word_list):
                    word = word_list[i]
                    vec = self.word2vec(word)
                    vector[i] = vec
            batch_label[batch_index] = label
            batch_data[batch_index] = vector
        return batch_label, batch_data

    # 将一句话转成向量
    def sentence2vector(self, sentence):
        # 去特殊字符
        sentence = re.sub(r'\W+', ' ', sentence).replace('_', ' ')
        # 分词
        cut = jieba.lcut(sentence)
        # 去空格
        cut = [item for item in cut if item != ' ']
        print(cut)
        vector = np.zeros(shape=(self.feature1_number, self.feature2_number))
        for i in range(self.feature1_number):
            if i < len(cut):
                word = cut[i]
                vec = self.word2vec(word)
                vector[i] = vec
        return vector

    # 将一句话转成测试数据
    def sentence2test_data(self, sentence):
        test_data = np.zeros(shape=(1, self.feature1_number, self.feature2_number))
        test_data[0] = self.sentence2vector(sentence)
        return test_data


if __name__ == '__main__':
    from data_loader import DataLoader

    feature1_number = 60
    feature2_number = 300
    data_loader = DataLoader()
    data_helper = DataHelper(feature1_number, feature2_number)
    print(data_helper.get_batch_label_and_vector(data_loader, 10))
