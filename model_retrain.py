import sys

import tensorflow.keras as keras

from model_train import train

argv = sys.argv
if len(argv) != 4:
    print('python3 model_retrain.py [model_filename] [batch_size] [epochs]')
    exit(0)

model_filename = argv[1]
batch_size = int(argv[2])
epochs = int(argv[3])

model = keras.models.load_model(model_filename)

train(model=model, save_filename=model_filename, batch_size=batch_size, epochs=epochs)
