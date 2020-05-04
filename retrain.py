import tensorflow.keras as keras

import config
from data_helper import DataHelper
from data_loader import DataLoader

# 模型
print('model loading...')
model_file_name = config.retrain_model_file_name
model = keras.models.load_model(model_file_name)
model.summary()

# 加载数据
print('data loading...')
data_loader = DataLoader()
print('vector loading...')
data_helper = DataHelper(config.feature1_number, config.feature2_number)

# 训练
batch_size = 32
model.fit_generator(
    generator=data_helper.generator(data_loader, batch_size),
    steps_per_epoch=data_loader.num_data // batch_size,
    epochs=10
)

# 测试
model.evaluate_generator(
    generator=data_helper.generator(data_loader, batch_size),
    steps=data_loader.num_data // batch_size,
)

# 保存
model.save(model_file_name)
print('model saved')
