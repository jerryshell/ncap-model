import sys

import tensorflow.keras as keras

from data_helper import DataHelper

argv = sys.argv
if len(argv) != 3:
    print('python3 model_evaluate.py [model_filename] [batch_size]')
    exit(0)

model_filename = argv[1]
batch_size = int(argv[2])

# 模型
print('model loading...')
model = keras.models.load_model(model_filename)
model.summary()

# 加载数据
print('data loading...')
data_helper = DataHelper()

# 数据生成器
test_data_generator = data_helper.test_data_generator(batch_size)

# 测试
model.evaluate(
    x=test_data_generator,
    steps=data_helper.test_data_count // batch_size,
)
