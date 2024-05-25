import tensorflow as tf
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import load_ren_data
import matplotlib.pyplot as plt

# Load data
noise_data_path = r"vzorky\poisson_intensity_0.1"
clean_data_path = r"vzorky\original"
x_data, y_data = load_ren_data.load_data(noise_data_path, clean_data_path)

plt.imshow(x_data[0])
plt.show()


x_data = np.expand_dims(x_data, -1)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Define your model architecture
def get_model(input_shape):
    model = keras.models.Sequential([
        keras.layers.Input((512,512,1)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        #keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'),  # Output layer for binary classification
        keras.layers.Reshape((512,512,1))
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the model
model = get_model(input_shape=x_train.shape[1:])

output = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# Use the trained model for inference
# predicted_mask = model.predict(new_noisy_image)
