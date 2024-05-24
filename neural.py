import tensorflow as tf
import keras
import numpy as np


import load_ren_data

#typy sumu
# poisson
# salt_pepper
# speckle

noise_data_path = r"vzorky\poisson_intensity_0.1"
clean_data_path = r"vzorky\original"

x_data, y_data = load_ren_data.load_data(noise_data_path, clean_data_path)

x_data = np.expand_dims(x_data, -1)
print("A")

def get_model():
  model = keras.models.Sequential()
  model.add(keras.layers.Input(shape=(512,512, 1)))
  model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(keras.layers.Conv2D(64, kernel_size=(5, 5), activation="relu"))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Dense(255, activation="softmax"))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model


EPOCHS = 10
model = get_model()
output = model.fit(x_data, y_data, epochs=EPOCHS, validation_split=0.1)




