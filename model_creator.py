import tensorflow.keras as keras

import model_config
from data_helper import DataHelper


def create_model_text_cnn(embedding_weights, embedding_trainable: bool):
    inputs = keras.layers.Input(shape=(model_config.word_count,))

    embedding = keras.layers.Embedding(
        input_dim=embedding_weights.shape[0],
        output_dim=embedding_weights.shape[1],
        weights=[embedding_weights],
        trainable=embedding_trainable,
        name='embedding',
    )(inputs)

    filters = 64
    kernel_sizes = [7, 6, 5]

    cnn1 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[0],
        name='cnn1',
    )(embedding)
    batch_normal1 = keras.layers.BatchNormalization(name='bn1')(cnn1)
    relu1 = keras.layers.ReLU(name='relu1')(batch_normal1)
    max_pool1 = keras.layers.MaxPooling1D(
        name='max_pool1',
    )(relu1)

    cnn2 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[1],
        name='cnn2',
    )(embedding)
    batch_normal2 = keras.layers.BatchNormalization(name='bn2')(cnn2)
    relu2 = keras.layers.ReLU(name='relu2')(batch_normal2)
    max_pool2 = keras.layers.MaxPooling1D(
        name='max_pool2',
    )(relu2)

    cnn3 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[2],
        name='cnn3',
    )(embedding)
    batch_normal3 = keras.layers.BatchNormalization(name='bn3')(cnn3)
    relu3 = keras.layers.ReLU()(batch_normal3)
    max_pool3 = keras.layers.MaxPooling1D(
        name='max_pool3',
    )(relu3)

    concatenate = keras.layers.Concatenate(axis=1)([max_pool1, max_pool2, max_pool3])

    flatten = keras.layers.Flatten()(concatenate)

    outputs = keras.layers.Dense(units=2, activation='softmax')(flatten)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )
    return model


if __name__ == '__main__':
    print('data loading...')
    data_helper = DataHelper()

    model = create_model_text_cnn(
        embedding_weights=data_helper.idx2vec,
        embedding_trainable=False
    )
    model.summary()
