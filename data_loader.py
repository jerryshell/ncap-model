import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self):
        # 加载数据
        data = pd.read_csv('./data/simplifyweibo_4_moods_cut.csv')
        print(data.shape)
        print(data.describe())

        # 打乱数据集
        self.data = np.array(data)
        np.random.shuffle(self.data)

        # 记录数据集的最大数量
        self.num_data = len(self.data)

        # 迭代索引
        self.next_index = 0

    def next(self):
        # 0 为正面情感，其他为负面情感转为 1
        item = self.data[self.next_index]
        y = item[0]
        if y > 0:
            y = 1
        x = item[1]
        # next_index 自增
        self.next_index += 1
        # 使用完一轮之后再次打乱数据，并重置 next_index
        if self.next_index == self.num_data:
            np.random.shuffle(self.data)
            self.next_index = 0
        return y, x


if __name__ == '__main__':
    data_loader = DataLoader()
    for i in range(10):
        print('---')
        print(data_loader.next())
