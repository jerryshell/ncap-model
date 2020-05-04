import tensorflow.keras as keras

import config
from train import train

print('model loading...')
model_file_name = config.retrain_model_filename
model = keras.models.load_model(model_file_name)

train(model, model_file_name)
model.summary()
