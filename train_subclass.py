import tensorflow as tf
import tensorflow.keras as keras

import config
from data_helper import DataHelper
from data_loader import DataLoader
from models import TextCNN

batch_size = 10

# 数据加载工具
print('data loading...')
data_loader = DataLoader()
print('vector loading...')
data_helper = DataHelper(config.feature1_number, config.feature2_number)

# 模型
model = TextCNN()

# 优化器
optimizer = keras.optimizers.Adam()

# 迭代训练
# num_batch = data_helper.data_loader.num_train_data // batch_size
num_batch = 10000
for batch_index in range(num_batch):
    # 从 data_loader 中随机取出一批训练数据
    y, X = data_helper.get_batch_label_and_vector(data_loader, batch_size)
    # print('y', y)
    with tf.GradientTape() as tape:
        # 将这批数据送入模型，计算出模型的预测值
        y_pred = model(X)
        # print('y_pred', y_pred)
        # 将模型的预测值与真实值进行比较，计算损失函数（loss）
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        # print('loss', loss)
        loss = tf.reduce_mean(loss)
        print('batch_index %d / %d: loss %f' % (batch_index, num_batch, loss.numpy()))
    # 计算损失函数关于模型变量的导数
    grads = tape.gradient(loss, model.variables)
    # 将求出的导数值传入优化器，使用优化器的 apply_gradients 方法更新模型参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

# 测试
print('testing...')
print(model.predict(data_helper.sentence2test_data('我好开心')))
print(model.predict(data_helper.sentence2test_data('我很失望')))

num_test_data = 1000
test_batch_size = 1
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
for test_index in range(num_test_data):
    y, X = data_helper.get_batch_label_and_vector(data_loader, test_batch_size)
    y_pred = model.predict(X)
    sparse_categorical_accuracy.update_state(y_true=y, y_pred=y_pred)
    print("%s / %s test accuracy: %f" % (test_index, num_test_data, sparse_categorical_accuracy.result()))

# 保存
print('saving...')
keras.models.save_model(model, 'save/1')
