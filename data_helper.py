import numpy as np

import model_config
from data_loader import DataLoader


class DataHelper:
    def __init__(self):
        # 加载字典
        self.word2idx = {}
        with open('./data/sgns.wiki.bigram-char') as f:
            length, dim = f.readline().strip().split(' ')
            length = int(length)
            dim = int(dim)
            print('length %s dim %s' % (length, dim))

            self.idx2vec = np.zeros(shape=(length, model_config.feature2_count))

            max_idx = 1
            for line in f:
                line_split = line.strip().split(' ')
                # print(values)
                if len(line_split) <= 300:
                    continue

                word = line_split[0]
                vec = np.asarray(line_split[1:], dtype='float32')

                self.word2idx[word] = max_idx
                self.idx2vec[max_idx] = vec

                max_idx += 1

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

    def word_list2idx_list(self, word_list: list):
        idx_list = np.zeros(shape=(model_config.feature1_count,))
        for i in range(model_config.feature1_count):
            if i >= len(word_list):
                break

            word = word_list[i]
            idx = self.word2idx.get(word)
            if idx is None:
                idx = self.word2idx.get("UNKNOWN")
            idx_list[i] = idx

        return idx_list

    # data_generator 中读取 batch_size 个数据，并转换成向量返回
    def get_batch_idx_and_label(self, data_generator: iter, batch_size):
        # 初始化返回结果
        batch_label = np.zeros(shape=(batch_size,))
        batch_idx = np.zeros(shape=(batch_size, model_config.feature1_count))
        # 根据 batch_size 填充返回结果
        for batch_index in range(batch_size):
            # 从 data_generator 中读取下一个数据
            label, sentence = next(data_generator)
            # 0 为正面情感，其他为负面情感转为 1
            label = 0 if label == 0 else 1
            # 分词，获得词语列表
            word_list = sentence.split(' ')
            # 把词语列表转换成向量
            idx = self.word_list2idx_list(word_list)
            # 填充返回结果
            batch_label[batch_index] = label
            batch_idx[batch_index] = idx
        return batch_idx, batch_label

    # 训练数据生成器
    def train_data_generator(self, batch_size):
        while True:
            yield self.get_batch_idx_and_label(self.raw_train_data_generator, batch_size)

    # 验证数据生成器
    def validation_data_generator(self, batch_size):
        while True:
            yield self.get_batch_idx_and_label(self.raw_validation_data_generator, batch_size)

    # 测试数据生成器
    def test_data_generator(self, batch_size):
        while True:
            yield self.get_batch_idx_and_label(self.raw_test_data_generator, batch_size)

    def sentence2idx(self, sentence):
        result = np.zeros(shape=(1, model_config.feature1_count))
        word_list = sentence.split(' ')
        result[0] = self.word_list2idx_list(word_list)
        return result


if __name__ == '__main__':
    DataHelper()
