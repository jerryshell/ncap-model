import tensorflow as tf


class TextCNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=48)

        self.conv2 = tf.keras.layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=47)

        self.conv3 = tf.keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling1D(pool_size=46)

        self.flat = tf.keras.layers.Flatten()
        self.drop = tf.keras.layers.Dropout(0.2)
        self.out = tf.keras.layers.Dense(2, activation='softmax')

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
