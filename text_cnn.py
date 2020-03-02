import tensorflow as tf
import tensorflow.keras as keras


class TextCNN(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')
        self.pool1 = keras.layers.MaxPooling1D(pool_size=48)

        self.conv2 = keras.layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')
        self.pool2 = keras.layers.MaxPooling1D(pool_size=47)

        self.conv3 = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')
        self.pool3 = keras.layers.MaxPooling1D(pool_size=46)

        self.flat = keras.layers.Flatten()
        self.drop = keras.layers.Dropout(0.2)
        self.out = keras.layers.Dense(2, activation='softmax')

    @tf.function
    def call(self, inputs, **kwargs):
        cnn1 = self.conv1(inputs)
        cnn1 = self.pool1(cnn1)

        cnn2 = self.conv2(inputs)
        cnn2 = self.pool2(cnn2)

        cnn3 = self.conv3(inputs)
        cnn3 = self.pool3(cnn3)

        merge = tf.concat([cnn1, cnn2, cnn3], axis=-1)

        flat = self.flat(merge)
        drop = self.drop(flat)
        out = self.out(drop)
        return out


def create_model(feature1_number, feature2_number):
    inputs = keras.layers.Input(shape=(feature1_number, feature2_number))

    cnn1 = keras.layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(inputs)
    cnn1 = keras.layers.MaxPooling1D(pool_size=48)(cnn1)

    cnn2 = keras.layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(inputs)
    cnn2 = keras.layers.MaxPooling1D(pool_size=47)(cnn2)

    cnn3 = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(inputs)
    cnn3 = keras.layers.MaxPooling1D(pool_size=46)(cnn3)

    cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = keras.layers.Flatten()(cnn)
    drop = keras.layers.Dropout(0.2)(flat)
    outputs = keras.layers.Dense(4, activation='softmax')(drop)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def create_model2(feature1_number, feature2_number):
    inputs = keras.layers.Input(shape=(feature1_number, feature2_number))
    flat = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(64, activation='relu')(flat)
    x = keras.layers.Dense(64, activation='relu')(x)
    outputs = keras.layers.Dense(2, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def create_model3(feature1_number, feature2_number):
    inputs = keras.layers.Input(shape=(feature1_number, feature2_number))

    lstm = tf.keras.layers.RNN(
        tf.keras.layers.LSTMCell(128)
    )(inputs)

    btn = tf.keras.layers.BatchNormalization()(lstm)

    outputs = keras.layers.Dense(4, activation='softmax')(btn)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='sgd',
        metrics=['accuracy']
    )
    return model
