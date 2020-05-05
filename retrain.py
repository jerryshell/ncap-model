import sys

import tensorflow.keras as keras

from train import train

argv = sys.argv
if len(argv) != 4:
    print('python3 retrain.py [model_file_name] [batch_size] [epochs]')
    exit(0)

model_file_name = argv[1]
batch_size = int(argv[2])
epochs = int(argv[3])

model = keras.models.load_model(model_file_name)

train(model=model, save_filename=model_file_name, batch_size=batch_size, epochs=epochs)
