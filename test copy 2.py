import tensorflow as tf
from keras.layers import Input, Conv2D, Concatenate, Dropout, MaxPool2D, Activation, GlobalAveragePooling2D, Dense, Multiply, AveragePooling2D, UpSampling2D, Add, BatchNormalization
from keras.models import Model
from keras.activations import softmax
from keras import regularizers
import numpy as np
from sklearn.model_selection import train_test_split
import load_ren_data

class Convolutional_block(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same')
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same')
        self.bn_2 = BatchNormalization()

    def call(self, X):
        X_1 = self.conv_1(X)
        X_1 = self.bn_1(X_1)
        X_1 = Activation('relu')(X_1)
        X_2 = self.conv_2(X_1)
        X_2 = self.bn_2(X_2)
        X_2 = Activation('relu')(X_2)
        return X_2

class Channel_attention(tf.keras.layers.Layer):
    def __init__(self, C=32, **kwargs):
        super().__init__(**kwargs)
        self.C = C
        self.gap = GlobalAveragePooling2D()
        self.dense_sigmoid = Dense(units=self.C, activation='sigmoid')
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'C': self.C})
        return config

    def call(self, X):
        v = self.gap(X)
        mu = self.dense_sigmoid(v)
        U_out = Multiply()([X, mu])
        return U_out

class Avg_pool_Unet_Upsample_msfe(tf.keras.layers.Layer):
    def __init__(self, avg_pool_size, upsample_rate, **kwargs):
        super().__init__(**kwargs)
        self.avg_pool_size = avg_pool_size
        self.upsample_rate = upsample_rate
        self.avg_pool = AveragePooling2D(pool_size=avg_pool_size, padding='same')

        self.conv_down_lst = [Conv2D(filters=32, kernel_size=[3, 3], activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2=0.001)) for _ in range(2)]
        self.bn_down_lst = [BatchNormalization() for _ in range(2)]
        
        self.conv_up_lst = [Conv2D(filters=32, kernel_size=[3, 3], activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2=0.001)) for _ in range(2)]
        self.bn_up_lst = [BatchNormalization() for _ in range(2)]
        
        self.conv_3 = Conv2D(filters=3, kernel_size=[1, 1])
        self.pooling_unet = MaxPool2D(pool_size=[2, 2], padding='same')
        self.upsample = UpSampling2D(upsample_rate, interpolation='bilinear')
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'avg_pool_size': self.avg_pool_size, 'upsample_rate': self.upsample_rate})
        return config

    def upsample_and_concat(self, x1, x2):
        deconv = UpSampling2D()(x1)
        return Concatenate()([deconv, x2])

    def unet(self, input):
        conv1 = input
        for c, bn in zip(self.conv_down_lst, self.bn_down_lst):
            conv1 = c(conv1)
            conv1 = bn(conv1)
        
        pool1 = self.pooling_unet(conv1)
        for c, bn in zip(self.conv_up_lst, self.bn_up_lst):
            conv1 = c(conv1)
            conv1 = bn(conv1)
        
        conv2 = self.conv_3(conv1)
        return conv2

    def call(self, X):
        avg_pool = self.avg_pool(X)
        unet = self.unet(avg_pool)
        upsample = self.upsample(unet)
        return upsample

class Multi_scale_feature_extraction(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.msfe_4 = Avg_pool_Unet_Upsample_msfe(avg_pool_size=4, upsample_rate=4)
        self.msfe_2 = Avg_pool_Unet_Upsample_msfe(avg_pool_size=2, upsample_rate=2)
        self.msfe_1 = Avg_pool_Unet_Upsample_msfe(avg_pool_size=1, upsample_rate=1)

    def call(self, X):
        up_sample_4 = self.msfe_4(X)
        up_sample_2 = self.msfe_2(X)
        up_sample_1 = self.msfe_1(X)
        msfe_out = Concatenate()([X, up_sample_4, up_sample_2, up_sample_1])
        return msfe_out

class Kernel_selecting_module(tf.keras.layers.Layer):
    def __init__(self, C=21, **kwargs):
        super().__init__(**kwargs)
        self.C = C
        self.c_3 = Conv2D(filters=self.C, kernel_size=(3,3), strides=1, padding='same', kernel_regularizer=regularizers.l2(l2=0.001))
        self.gap = GlobalAveragePooling2D()
        self.dense_c = Dense(units=self.C)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'C': self.C})
        return config

    def call(self, X):
        X_1 = self.c_3(X)
        v_gap = self.gap(X_1)
        v_gap = tf.reshape(v_gap, [-1, 1, 1, self.C])
        alpha = self.dense_c(v_gap)
        select = Multiply()([X_1, alpha])
        return select

# Input Layer
input_layer = Input(shape=(512, 512, 1))

# Convolutional Block
conv_block = Convolutional_block()(input_layer)
drop1 = Dropout(0.1)(conv_block)
# Channel Attention
channel_attention = Channel_attention()(drop1)

# Conv2D Layer
conv2d_4 = Conv2D(1, (3, 3), padding='same', activation='relu')(channel_attention)
drop2 = Dropout(0.2)(conv2d_4)

# Concatenate
concat_input_drop2 = Concatenate()([input_layer, drop2])

# Multi Scale Feature Extraction
multi_scale_feature_extraction = Multi_scale_feature_extraction()(concat_input_drop2)

# Kernel Selecting Module
kernel_selecting_module = Kernel_selecting_module()(multi_scale_feature_extraction)

# Final Conv2D Layer
output_layer = Conv2D(1, (3, 3), padding='same', activation='relu')(kernel_selecting_module)

# Create Model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mean_squared_error", metrics=["accuracy"])

# Summary
model.summary()

noise_data_path = r"vzorky\poisson_intensity_0.1"
clean_data_path = r"vzorky\original"

x_data, y_data  = load_ren_data.load_data(noise_data_path, clean_data_path)

x_data = np.expand_dims(x_data, -1)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

model.fit(x_train, y_train, epochs=10, batch_size=16, validation_data=(x_test, y_test))
