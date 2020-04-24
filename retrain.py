import tensorflow.keras as keras

import config
from data_helper import DataHelper
from data_loader import DataLoader

# 数据加载工具
print('data loading...')
data_loader = DataLoader()
print('vector loading...')
data_helper = DataHelper(config.feature1_number, config.feature2_number)

# 模型
print('model loading...')
model_file_name = 'retrain.text_cnn_separable.2.80.h5'
model = keras.models.load_model(model_file_name)
model.summary()

# 加载数据
epoch = 5
batch_size = 32
num_data = data_loader.num_data
num_batch = num_data // batch_size * epoch
for batch_index in range(num_batch):
    print('%s / %s' % (batch_index, num_batch))
    y, X = data_helper.get_batch_label_and_vector(data_loader, batch_size)
    model.fit(X, y, batch_size=batch_size)
    if batch_index % 10 == 0:
        model.save(model_file_name)
        print('model saved')

# 测试
num_data = int(data_loader.num_data / 10)
num_batch = num_data // batch_size
for batch_index in range(num_batch):
    print('%s / %s' % (batch_index, num_batch))
    y, X = data_helper.get_batch_label_and_vector(data_loader, batch_size)
    model.evaluate(X, y)

# 保存
model.save(model_file_name)
print('model saved')
