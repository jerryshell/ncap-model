import re

import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self):
        data = pd.read_csv('./data/simplifyweibo_4_moods.csv')
        self.train_label = data['label'].values
        self.train_data = data['review'].values
        self.num_train_data = self.train_data.shape[0]

    def next(self):
        index = np.random.randint(0, self.num_train_data)
        label = 0 if 0 == self.train_label[index] else 1
        data = re.sub(r'\W+', ' ', self.train_data[index]).replace('_', ' ')
        return label, data
