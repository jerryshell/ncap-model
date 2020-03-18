import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self):
        # 加载数据
        data = pd.read_csv('./data/simplifyweibo_4_moods_cut.csv')
        print(data.shape)
        print(data.describe())

        # 打乱数据集
        data = np.array(data)
        np.random.shuffle(data)

        # 拆分数据和标签
        self.y = data[:, 0]
        self.x = data[:, 1]

        # 记录数据集的最大数量
        self.num_data = len(self.x)

        # 迭代索引
        self.next_index = 0

    def next(self):
        # 0 为正面情感，其他为负面情感转为 1
        y = self.y[self.next_index]
        if y > 0:
            y = 1
        x = self.x[self.next_index]
        # next_index 自增
        self.next_index = (self.next_index + 1) % self.num_data
        return y, x


if __name__ == '__main__':
    data_loader = DataLoader()
    for i in range(10):
        print('---')
        print(data_loader.next())
