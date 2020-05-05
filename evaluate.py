import tensorflow.keras as keras

import config
from data_helper import DataHelper
from data_loader import DataLoader

model_file_name = 'text_cnn_separable.2.h5'
batch_size = 32

# 数据加载工具
print('data loading...')
data_loader = DataLoader()
print('vector loading...')
data_helper = DataHelper(config.feature1_number, config.feature2_number)

# 模型
print('model loading...')
model = keras.models.load_model(model_file_name)
model.summary()

model.evaluate(
    x=data_helper.generator(data_loader, batch_size),
    steps=data_loader.num_data // batch_size,
)
