import itertools

import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self):
        # 加载数据
        data = pd.read_csv('./data/simplifyweibo_4_moods_cut.csv')
        print('data.shape', data.shape)
        print(data.describe())

        # pandas 转 numpy
        data = data.to_numpy()
        data_count = len(data)

        # 打乱数据集
        np.random.shuffle(data)

        # 切分训练集、验证集、测试集 8:1:1
        self.train_data, self.validation_data, self.test_data = np.vsplit(
            data,
            [int(data_count * 0.8), int(data_count * 0.9)]
        )
        print('train_data.shape', self.train_data.shape)
        print('validation_data.shape', self.validation_data.shape)
        print('test_data.shape', self.test_data.shape)

        # 记录训练集、验证集、测试集的大小
        self.train_data_count = len(self.train_data)
        self.validation_data_count = len(self.validation_data)
        self.test_data_count = len(self.test_data)
        print('train_data_count', self.train_data_count)
        print('validation_data_count', self.validation_data_count)
        print('test_data_count', self.test_data_count)

    def train_data_generator(self):
        return itertools.cycle(self.train_data)

    def validation_data_generator(self):
        return itertools.cycle(self.validation_data)

    def test_data_generator(self):
        return itertools.cycle(self.test_data)


if __name__ == '__main__':
    data_loader = DataLoader()

    train_data_generator = data_loader.train_data_generator()
    validation_data_generator = data_loader.validation_data_generator()
    test_data_generator = data_loader.test_data_generator()

    print(next(train_data_generator))
    print(next(train_data_generator))
    print('---')
    print(next(validation_data_generator))
    print(next(validation_data_generator))
    print('---')
    print(next(test_data_generator))
    print(next(test_data_generator))
