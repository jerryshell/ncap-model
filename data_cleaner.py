import pandas as pd

data = pd.read_csv('./data/simplifyweibo_4_moods_cut.csv')
print(data.isnull().any())
data = data.dropna()
print(data.isnull().any())
data.to_csv('./data/simplifyweibo_4_moods_cut.csv', index=False)
