import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self):
        data = pd.read_csv('./data/simplifyweibo_4_moods_cut.csv')
        # data = pd.concat([data, data[199496:], data[199496:], data[199496:]])
        print(data.shape)
        print(data.describe())
        self.train_label = data['label'].values
        self.train_data = data['review'].values
        self.num_train_data = self.train_data.shape[0]

    def next(self):
        index = np.random.randint(0, self.num_train_data)
        # if index <= 199496:
        #     index = np.random.randint(0, self.num_train_data)
        label = 0 if 0 == self.train_label[index] else 1
        return label, self.train_data[index]
