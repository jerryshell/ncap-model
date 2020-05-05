import sys

import tensorflow.keras as keras

import model_config
from data_helper import DataHelper
from data_loader import DataLoader

argv = sys.argv
if len(argv) != 3:
    print('python3 model_evaluate.py [model_file_name] [batch_size]')
    exit(0)

model_file_name = argv[1]
batch_size = int(argv[2])

# 数据加载工具
print('data loading...')
data_loader = DataLoader()
print('vector loading...')
data_helper = DataHelper(model_config.feature1_count, model_config.feature2_count)

# 模型
print('model loading...')
model = keras.models.load_model(model_file_name)
model.summary()

model.evaluate(
    x=data_helper.generator(data_loader, batch_size),
    steps=data_loader.num_data // batch_size,
)
