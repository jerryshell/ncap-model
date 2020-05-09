from datetime import datetime

import tensorflow.keras as keras

from data_helper import DataHelper


def train(
        data_helper: DataHelper,
        model: keras.Model,
        save_filename: str,
        batch_size=32,
        epochs=10
):
    # 模型信息
    model.summary()

    # 训练数据生成器
    train_data_generator = data_helper.train_data_generator(batch_size)
    # 验证数据生成器
    validation_data_generator = data_helper.validation_data_generator(batch_size)
    # 测试数据生成器
    test_data_generator = data_helper.test_data_generator(batch_size)

    # 训练
    model.fit(
        x=train_data_generator,
        steps_per_epoch=data_helper.train_data_count // batch_size,
        validation_data=validation_data_generator,
        validation_steps=data_helper.validation_data_count // batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[
            # 配置 tensorboard，将训练过程可视化，方便调参，tensorboard --logdir logs/fit
            keras.callbacks.TensorBoard(
                log_dir='logs/fit/' + datetime.now().strftime('%Y%m%d-%H%M%S'),
                histogram_freq=1
            ),
            # 定时保存模型
            keras.callbacks.ModelCheckpoint(
                filepath=save_filename,
                monitor='val_loss',
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                save_freq='epoch'
            )
        ],
    )

    # 测试
    model.evaluate(
        x=test_data_generator,
        steps=data_helper.test_data_count // batch_size,
    )


if __name__ == '__main__':
    import sys
    import model_creator

    print(sys.argv)
    if len(sys.argv) != 4:
        print('python3 model_train.py <batch_size> <epochs> <embedding_trainable[true/false]>')
        exit(0)

    batch_size = int(sys.argv[1])
    epochs = int(sys.argv[2])
    embedding_trainable = True if sys.argv[3].lower() == 'true' else False
    print('batch_size %s epochs %s embedding_trainable %s' % (batch_size, epochs, embedding_trainable))

    # 加载数据
    print('data loading...')
    data_helper = DataHelper()

    # 重新训练一个新模型
    model = model_creator.create_model_text_cnn(
        embedding_weights=data_helper.idx2vec,
        embedding_trainable=embedding_trainable
    )
    train(
        data_helper=data_helper,
        model=model,
        save_filename='text_cnn.2.h5',
        batch_size=batch_size,
        epochs=epochs
    )
