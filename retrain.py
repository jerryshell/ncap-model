import tensorflow.keras as keras

from data_helper import DataHelper
from data_loader import DataLoader

# 超参数
feature1_number = 60  # 句子分成多少个词语，多余截断，不够补 0
feature2_number = 300  # 每个词语的向量

# 数据加载工具
print('data loading...')
data_loader = DataLoader()
print('vector loading...')
data_helper = DataHelper(feature1_number, feature2_number)

# 模型
print('model loading...')
model_file_name = 'text_cnn'
model = keras.models.load_model(model_file_name + '.h5')

# 加载数据
epoch = 10
batch_size = 100
num_data = data_loader.num_train_data
num_batch = num_data // batch_size * epoch
for batch_index in range(num_batch):
    print('%s / %s' % (batch_index, num_batch))
    y, X = data_helper.get_batch_label_and_vector(data_loader, batch_size)
    model.fit(X, y)
    if batch_index % 10 == 0:
        print('model saved')
        model.save('text_cnn.h5')

# 测试
num_data = int(data_loader.num_train_data / 2)
num_batch = num_data // batch_size
for batch_index in range(num_batch):
    print('%s / %s' % (batch_index, num_batch))
    y, X = data_helper.get_batch_label_and_vector(data_loader, batch_size)
    model.evaluate(X, y)

# 保存
model.save('text_cnn.h5')
