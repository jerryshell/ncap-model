import tensorflow.keras as keras

import model_config
from data_helper import DataHelper


def create_model_text_cnn(embedding_weights, embedding_trainable: bool):
    inputs = keras.layers.Input(shape=(model_config.feature1_count,))

    embedding = keras.layers.Embedding(
        input_dim=embedding_weights.shape[0],
        output_dim=model_config.feature2_count,
        weights=[embedding_weights],
        trainable=embedding_trainable,
        name='embedding',
    )(inputs)

    filters = 128
    kernel_sizes = [2, 3, 4]
    padding = 'valid'

    cnn1 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[0],
        strides=1,
        padding=padding,
        activation='relu',
        name='cnn1',
    )(embedding)
    max_pool1 = keras.layers.MaxPooling1D(
        pool_size=2,
        name='max_pool1',
    )(cnn1)

    cnn2 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[1],
        strides=1,
        padding=padding,
        activation='relu',
        name='cnn2',
    )(embedding)
    max_pool2 = keras.layers.MaxPooling1D(
        pool_size=2,
        name='max_pool2',
    )(cnn2)

    cnn3 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[2],
        strides=1,
        padding=padding,
        activation='relu',
        name='cnn3',
    )(embedding)
    max_pool3 = keras.layers.MaxPooling1D(
        pool_size=2,
        name='max_pool3',
    )(cnn3)

    concatenate = keras.layers.Concatenate(axis=1)([max_pool1, max_pool2, max_pool3])

    flatten = keras.layers.Flatten()(concatenate)

    dropout = keras.layers.Dropout(0.5)(flatten)

    outputs = keras.layers.Dense(2, activation='softmax')(dropout)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )
    return model


def create_model_lstm(embedding_weights, embedding_trainable: bool):
    inputs = keras.layers.Input(shape=(model_config.feature1_count,))

    embedding = keras.layers.Embedding(
        input_dim=embedding_weights.shape[0],
        output_dim=model_config.feature2_count,
        weights=[embedding_weights],
        trainable=embedding_trainable,
        name='embedding',
    )(inputs)

    lstm = keras.layers.LSTM(
        units=32,
        name='lstm',
    )(embedding)

    dropout = keras.layers.Dropout(0.5)(lstm)

    outputs = keras.layers.Dense(2, activation='softmax')(dropout)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )
    return model


if __name__ == '__main__':
    # 加载数据
    print('data loading...')
    data_helper = DataHelper()

    model = create_model_lstm(
        embedding_weights=data_helper.idx2vec,
        embedding_trainable=False
    )
    model.summary()
