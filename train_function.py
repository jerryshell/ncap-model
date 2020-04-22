import config
from data_helper import DataHelper
from data_loader import DataLoader
from models import create_model_text_cnn_separable

# 数据加载工具
print('data loading...')
data_loader = DataLoader()
print('vector loading...')
data_helper = DataHelper(config.feature1_number, config.feature2_number)

# 模型
model = create_model_text_cnn_separable(config.feature1_number, config.feature2_number)

# 加载数据
epoch = 10
batch_size = 64
num_data = data_loader.num_data
num_batch = num_data // batch_size * epoch
for batch_index in range(num_batch):
    print('%s / %s' % (batch_index, num_batch))
    y, X = data_helper.get_batch_label_and_vector(data_loader, batch_size)
    model.fit(X, y, batch_size=batch_size)

# 测试
num_data = int(data_loader.num_data / 2)
num_batch = num_data // batch_size
for batch_index in range(num_batch):
    print('%s / %s' % (batch_index, num_batch))
    y, X = data_helper.get_batch_label_and_vector(data_loader, batch_size)
    model.evaluate(X, y)

# 保存
model.save('text_cnn_separable.2.h5')
