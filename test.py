import tensorflow.keras as keras

import config
from data_helper import DataHelper

# 加载模型
model = keras.models.load_model("save/1")

# 数据工具
data_helper = DataHelper(config.feature1_number, config.feature2_number)

while True:
    user_input = input('>>> ')
    test_data = data_helper.sentence2test_data(user_input)
    result = model.predict(test_data)
    a = result[0][0]
    b = result[0][1]
    print('正面言论概率：%s%% 负面言论概率：%s%%' % (a * 100, b * 100))
