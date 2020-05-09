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

    filters = 512
    kernel_sizes = [6, 5, 4]

    cnn1 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[0],
        activation='relu',
        name='cnn1',
    )(embedding)
    max_pool1 = keras.layers.MaxPooling1D(
        name='max_pool1',
    )(cnn1)

    cnn2 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[1],
        activation='relu',
        name='cnn2',
    )(embedding)
    max_pool2 = keras.layers.MaxPooling1D(
        name='max_pool2',
    )(cnn2)

    cnn3 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[2],
        activation='relu',
        name='cnn3',
    )(embedding)
    max_pool3 = keras.layers.MaxPooling1D(
        name='max_pool3',
    )(cnn3)

    concatenate = keras.layers.Concatenate(axis=1)([max_pool1, max_pool2, max_pool3])

    flatten = keras.layers.Flatten()(concatenate)

    dropout = keras.layers.Dropout(rate=0.5)(flatten)

    outputs = keras.layers.Dense(units=2, activation='softmax')(dropout)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy'],
    )
    return model


if __name__ == '__main__':
    # 加载数据
    print('data loading...')
    data_helper = DataHelper()

    model = create_model_text_cnn(
        embedding_weights=data_helper.idx2vec,
        embedding_trainable=False
    )
    model.summary()
