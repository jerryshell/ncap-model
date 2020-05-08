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

    dropout = keras.layers.Dropout(rate=0.5)(flatten)

    outputs = keras.layers.Dense(units=2, activation='softmax')(dropout)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )
    return model


def create_model_cnn(embedding_weights, embedding_trainable: bool):
    inputs = keras.layers.Input(shape=(model_config.feature1_count,))

    embedding = keras.layers.Embedding(
        input_dim=embedding_weights.shape[0],
        output_dim=model_config.feature2_count,
        weights=[embedding_weights],
        trainable=embedding_trainable,
        name='embedding',
    )(inputs)

    conv1 = keras.layers.SeparableConv1D(
        filters=16,
        kernel_size=5,
        activation='relu',
        # kernel_regularizer=keras.regularizers.l2(),
    )(embedding)
    max_pool = keras.layers.MaxPooling1D(pool_size=5)(conv1)

    # dropout1 = keras.layers.Dropout(0.5)(max_pool)

    conv2 = keras.layers.SeparableConv1D(
        filters=16,
        kernel_size=5,
        activation='relu',
        # kernel_regularizer=keras.regularizers.l2(),
    )(max_pool)
    global_max_pool = keras.layers.GlobalMaxPooling1D()(conv2)

    dropout2 = keras.layers.Dropout(0.5)(global_max_pool)

    outputs = keras.layers.Dense(units=2, activation='softmax')(dropout2)

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

    model = create_model_cnn(
        embedding_weights=data_helper.idx2vec,
        embedding_trainable=False
    )
    model.summary()
