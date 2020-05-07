import tensorflow.keras as keras


def create_model(feature1_number, feature2_number):
    inputs = keras.layers.Input(shape=(feature1_number, feature2_number))

    filters = 128
    kernel_sizes = [3, 4, 5]
    padding = 'valid'

    cnn1 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[0],
        strides=1,
        padding=padding,
        activation='relu',
    )(inputs)
    cnn1 = keras.layers.MaxPooling1D(
        pool_size=4,
    )(cnn1)

    cnn2 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[1],
        strides=1,
        padding=padding,
        activation='relu',
    )(inputs)
    cnn2 = keras.layers.MaxPooling1D(
        pool_size=4,
    )(cnn2)

    cnn3 = keras.layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_sizes[2],
        strides=1,
        padding=padding,
        activation='relu'
    )(inputs)
    cnn3 = keras.layers.MaxPooling1D(
        pool_size=4,
    )(cnn3)

    cnn = keras.layers.Concatenate(axis=-1)([cnn1, cnn2, cnn3])

    flat = keras.layers.Flatten()(cnn)

    drop = keras.layers.Dropout(0.5)(flat)

    outputs = keras.layers.Dense(2, activation='softmax')(drop)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


if __name__ == '__main__':
    import model_config

    model = create_model(model_config.feature1_count, model_config.feature2_count)
    model.summary()
