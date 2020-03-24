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
model_file_name = 'text_cnn_separable'
model = keras.models.load_model(model_file_name + '.h5')

# 读取数据
y, X = data_helper.get_batch_label_and_vector(
    data_loader,
    data_loader.num_data
)

# 评估模型
model.evaluate(X, y)
