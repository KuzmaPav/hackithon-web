import tensorflow as tf
from keras.layers import Input, Conv2D, Concatenate, Dropout
from keras.models import Model

import numpy as np
from sklearn.model_selection import train_test_split

import load_ren_data

# Define custom layers as per the diagram (assuming custom layers are implemented elsewhere)
class Convolutional_block(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(Convolutional_block, self).__init__()
        self.conv = Conv2D(filters, (3, 3), padding='same', activation='relu')

    def call(self, inputs):
        return self.conv(inputs)

class Channel_attention(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(Channel_attention, self).__init__()
        self.conv = Conv2D(filters, (1, 1), padding='same', activation='relu')

    def call(self, inputs):
        return self.conv(inputs)

class Multi_scale_feature_extraction(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(Multi_scale_feature_extraction, self).__init__()
        self.conv = Conv2D(filters, (3, 3), padding='same', activation='relu')

    def call(self, inputs):
        return self.conv(inputs)

class Kernel_selecting_module(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(Kernel_selecting_module, self).__init__()
        self.conv = Conv2D(filters, (1, 1), padding='same', activation='relu')

    def call(self, inputs):
        return self.conv(inputs)

# Input Layer
input_layer = Input(shape=(512, 512, 1))

# Convolutional Block
conv_block = Convolutional_block(64)(input_layer)

# Channel Attention
channel_attention = Channel_attention(64)(conv_block)

# Conv2D Layer
conv2d_4 = Conv2D(1, (3, 3), padding='same', activation='relu')(channel_attention)
drop1 = Dropout(0.2)(conv2d_4)

# Concatenate
concatenate = Concatenate()([input_layer, drop1])

# Multi Scale Feature Extraction
multi_scale_feature_extraction = Multi_scale_feature_extraction(21)(concatenate)

# Kernel Selecting Module
kernel_selecting_module = Kernel_selecting_module(21)(multi_scale_feature_extraction)

# Final Conv2D Layer
output_layer = Conv2D(1, (3, 3), padding='same', activation='relu')(kernel_selecting_module)

# Create Model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile("adam", "MSE", metrics=["accuracy"])

# Summary
model.summary()



noise_data_path = r"vzorky\poisson_intensity_0.1"
clean_data_path = r"vzorky\original"

x_data, y_data  = load_ren_data.load_data(noise_data_path, clean_data_path)

x_data = np.expand_dims(x_data, -1)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

model.fit(x_train, y_train, epochs=5)
