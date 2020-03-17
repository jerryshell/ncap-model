import re

import jieba
import pandas as pd

data = pd.read_csv('./data/simplifyweibo_4_moods.csv')
review = data['review'].values
index = 0
for item in review:
    print('%s / %s' % (index, review.shape[0]))
    # 删除特殊字符
    item = re.sub(r'\W+', ' ', item).replace('_', ' ')
    print(item)
    # 分词
    cut = jieba.lcut(item)
    # 删除空格
    cut = [item for item in cut if item != ' ']
    print(cut)
    # 更新分词后的数据
    review.put(index, ' '.join(cut))
    index += 1

print('----------')
print(review)
print('----------')
print(data)

# 保存
data.to_csv('./data/simplifyweibo_4_moods_cut.csv', index=False)
