import tensorflow.keras as keras

from train import train

model_file_name = 'retrain.text_cnn_separable.2.80.h5'

print('model loading...')
model = keras.models.load_model(model_file_name)

train(model=model, save_filename=model_file_name, batch_size=32, epochs=10)
model.summary()
