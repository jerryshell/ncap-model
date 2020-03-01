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
            cut = jieba.lcut(data)
            cut = [item for item in cut if item != ' ']
            # print(cut)
            # print(len(cut))
            vector = np.zeros(shape=(self.feature1_number, self.feature2_number))
            for i in range(self.feature1_number):
                if i < len(cut):
                    word = cut[i]
                    vec = self.word2vec(word)
                    vector[i] = vec
            batch_label[batch_index] = label
            batch_data[batch_index] = vector
        return batch_label, batch_data

    def get_vector_by_str(self, str):
        cut = jieba.lcut(str)
        cut = [item for item in cut if item != ' ']
        vector = np.zeros(shape=(self.feature1_number, self.feature2_number))
        for i in range(self.feature1_number):
            if i < len(cut):
                word = cut[i]
                vec = self.word2vec(word)
                vector[i] = vec
        return vector

    def get_test_data_by_str(self, str):
        batch_data = np.zeros(shape=(1, self.feature1_number, self.feature2_number))
        batch_data[0] = self.get_vector_by_str(str)
        return batch_data
