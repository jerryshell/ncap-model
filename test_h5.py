import tensorflow.keras as keras

from data_helper import DataHelper

# 超参数
feature1_number = 60  # 句子分成多少个词语，多余截断，不够补 0
feature2_number = 300  # 每个词语的向量

# 加载模型
model = keras.models.load_model("save.h5")

# 数据工具
data_helper = DataHelper(feature1_number, feature2_number)

while True:
    user_input = input('>>> ')
    test_data = data_helper.sentence2test_data(user_input)
    result = model.predict(test_data)
    a = result[0][0] * 100
    b = result[0][1] * 100
    c = result[0][2] * 100
    d = result[0][3] * 100
    # print('正面言论概率：%s%% 负面言论概率：%s%%' % (a * 100, b * 100))
    print('喜悦概率：%.2f%%\n愤怒概率：%.2f%%\n厌恶概率：%.2f%%\n低落概率：%.2f%%' % (a, b, c, d))
