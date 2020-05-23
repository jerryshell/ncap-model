import tensorflow.keras as keras

import model_config
from data_helper import DataHelper


def create_model_text_cnn(embedding_weights, embedding_trainable: bool):
    inputs = keras.layers.Input(
        shape=(model_config.word_count,),
        name='inputs'
    )

    embedding = keras.layers.Embedding(
        input_dim=embedding_weights.shape[0],
        output_dim=embedding_weights.shape[1],
        weights=[embedding_weights],
        trainable=embedding_trainable,
        name='embedding',
    )(inputs)

    filters = 700
    kernel_sizes = [6, 5, 4, 3]

    cnn1 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[0],
        name='cnn1',
    )(embedding)
    relu1 = keras.layers.ReLU(name='relu1')(cnn1)
    max_pool1 = keras.layers.MaxPooling1D(
        name='max_pool1',
    )(relu1)

    cnn2 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[1],
        name='cnn2',
    )(embedding)
    relu2 = keras.layers.ReLU(name='relu2')(cnn2)
    max_pool2 = keras.layers.MaxPooling1D(
        name='max_pool2',
    )(relu2)

    cnn3 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[2],
        name='cnn3',
    )(embedding)
    relu3 = keras.layers.ReLU(name='relu3')(cnn3)
    max_pool3 = keras.layers.MaxPooling1D(
        name='max_pool3',
    )(relu3)

    cnn4 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[3],
        name='cnn4',
    )(embedding)
    relu4 = keras.layers.ReLU(name='relu4')(cnn4)
    max_pool4 = keras.layers.MaxPooling1D(
        name='max_pool4',
    )(relu4)

    concatenate = keras.layers.Concatenate(axis=1)([max_pool1, max_pool2, max_pool3, max_pool4])

    flatten = keras.layers.Flatten()(concatenate)

    dropout = keras.layers.Dropout(rate=0.5)(flatten)

    outputs = keras.layers.Dense(
        units=2,
        activation='softmax',
        name='outputs',
    )(dropout)

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
