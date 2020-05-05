import datetime

import tensorflow as tf
from tensorflow import keras

import config
from data_helper import DataHelper
from data_loader import DataLoader


def train(model: keras.Model, save_filename: str, batch_size=32, epochs=10):
    # 模型信息
    model.summary()

    # 加载数据
    print('data loading...')
    data_loader = DataLoader()
    print('vector loading...')
    data_helper = DataHelper(config.feature1_number, config.feature2_number)

    # 配置 tensorboard，将训练过程可视化，方便调参
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # 训练
    model.fit(
        x=data_helper.generator(data_loader, batch_size),
        steps_per_epoch=data_loader.num_data // batch_size,
        epochs=epochs,
        callbacks=[
            # tensorboard --logdir logs/fit
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
            # 定时保存模型
            tf.keras.callbacks.ModelCheckpoint(
                save_filename, monitor='accuracy', verbose=0, save_best_only=True,
                save_weights_only=False, mode='auto', save_freq='epoch'
            )
        ],
    )

    # 测试
    model.evaluate(
        x=data_helper.generator(data_loader, batch_size),
        steps=data_loader.num_data // batch_size,
    )


if __name__ == '__main__':
    import sys
    from models import create_model_text_cnn_separable

    print(sys.argv)
    if len(sys.argv) != 3:
        print('python3 train.py [batch_size] [epochs]')
        exit(0)

    batch_size = int(sys.argv[1])
    epochs = int(sys.argv[2])
    print('batch_size %s epochs %s' % (batch_size, epochs))

    # 重新训练一个新模型
    model = create_model_text_cnn_separable(config.feature1_number, config.feature2_number)
    train(
        model=model,
        save_filename='text_cnn_separable.2.h5',
        batch_size=batch_size,
        epochs=epochs
    )
