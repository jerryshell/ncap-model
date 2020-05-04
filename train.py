import datetime

import tensorflow as tf
from tensorflow import keras

import config
from data_helper import DataHelper
from data_loader import DataLoader
from models import create_model_text_cnn_separable


def train(model: keras.Model, save_filename: str, batch_size=32, epochs=10):
    # 模型信息
    model.summary()

    # 加载数据
    print('data loading...')
    data_loader = DataLoader()
    print('vector loading...')
    data_helper = DataHelper(config.feature1_number, config.feature2_number)

    # 配置 tensorboard，将训练过程可视化，方便调参 tensorboard --logdir logs/fit
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 训练
    model.fit(
        x=data_helper.generator(data_loader, batch_size),
        steps_per_epoch=data_loader.num_data // batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[tensorboard_callback],
        use_multiprocessing=True,
    )

    # 测试
    model.evaluate(
        x=data_helper.generator(data_loader, batch_size),
        steps=data_loader.num_data // batch_size,
        use_multiprocessing=True,
    )

    # 保存
    model.save(save_filename)
    print('model saved')


if __name__ == '__main__':
    # 重新训练一个新模型
    model = create_model_text_cnn_separable(config.feature1_number, config.feature2_number)
    train(model=model, save_filename=config.train_model_filename, batch_size=32, epochs=1)
