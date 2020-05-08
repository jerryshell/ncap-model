import tensorflow.keras as keras

import model_config
from data_helper import DataHelper


def create_model(emb_weights, emb_trainable: bool):
    inputs = keras.layers.Input(shape=(model_config.feature1_count,))

    emb = keras.layers.Embedding(
        input_dim=emb_weights.shape[0],
        output_dim=model_config.feature2_count,
        weights=[emb_weights],
        trainable=emb_trainable
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
    )(emb)
    cnn1 = keras.layers.MaxPooling1D(
        pool_size=2,
    )(cnn1)

    cnn2 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[1],
        strides=1,
        padding=padding,
        activation='relu',
    )(emb)
    cnn2 = keras.layers.MaxPooling1D(
        pool_size=2,
    )(cnn2)

    cnn3 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[2],
        strides=1,
        padding=padding,
        activation='relu',
    )(emb)
    cnn3 = keras.layers.MaxPooling1D(
        pool_size=2,
    )(cnn3)

    cnn = keras.layers.Concatenate(axis=-1)([cnn1, cnn2, cnn3])

    flat = keras.layers.Flatten()(cnn)

    drop = keras.layers.Dropout(0.5)(flat)

    outputs = keras.layers.Dense(2, activation='softmax')(drop)

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

    model = create_model(emb_weights=data_helper.idx2vec, emb_trainable=False)
    model.summary()
