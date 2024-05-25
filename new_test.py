import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import History

# Načtení obrázků a jejich příprava
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(directory, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (128, 128))  # Přizpůsobení velikosti obrázku
            images.append(img)
    return np.array(images)

# Adresáře s obrázky
poisson_dir_1 = "./vzorky/poisson_intensity_0.1"
poisson_dir_2 = "./vzorky/poisson_intensity_0.2"
poisson_dir_3 = "./vzorky/poisson_intensity_0.3"
salt_pepper_dir_1 = "./vzorky/salt_pepper_intensity_0.01"
salt_pepper_dir_2 = "./vzorky/salt_pepper_intensity_0.03"
salt_pepper_dir_3 = "./vzorky/salt_pepper_intensity_0.05"
speckle_dir_1 = "./vzorky/speckle_intensity_0.1"
speckle_dir_2 = "./vzorky/speckle_intensity_0.05"
speckle_dir_3 = "./vzorky/speckle_intensity_0.15"
combined_dir_1 = "./vzorky/salt_and_pepper_speckle_intensity_0.01_0.1"
combined_dir_2 = "./vzorky/salt_and_pepper_speckle_intensity_0.03_0.3"
combined_dir_3 = "./vzorky/salt_and_pepper_speckle_intensity_0.05_0.2"
clean_dir = "./vzorky/original"

# Načtení obrázků
poisson_images_1 = load_images(poisson_dir_1)
poisson_images_2 = load_images(poisson_dir_2)
poisson_images_3 = load_images(poisson_dir_3)

salt_pepper_images_1 = load_images(salt_pepper_dir_1)
salt_pepper_images_2 = load_images(salt_pepper_dir_2)
salt_pepper_images_3 = load_images(salt_pepper_dir_3)

speckle_images_1 = load_images(speckle_dir_1)
speckle_images_2 = load_images(speckle_dir_2)
speckle_images_3 = load_images(speckle_dir_3)

combined_images_1 = load_images(combined_dir_1)
combined_images_2 = load_images(combined_dir_2)
combined_images_3 = load_images(combined_dir_3)
clean_images = load_images(clean_dir)

# Spojení všech typů obrázků se šumem
noisy_images = np.concatenate((
    poisson_images_1, poisson_images_2, poisson_images_3,
    salt_pepper_images_1, salt_pepper_images_2, salt_pepper_images_3,
    speckle_images_1, speckle_images_2, speckle_images_3,
    combined_images_1, combined_images_2, combined_images_3
))

# Normalizace pixelů
noisy_images = noisy_images.astype('float32') / 255.0
clean_images = clean_images.astype('float32') / 255.0

# Vytvoření a trénování modelu
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])  # Přidání metriky accuracy

# Vytvoření historie pro ukládání metrik
history = History()

# Trénování modelu s použitím historie
model.fit(noisy_images, clean_images, epochs=10, batch_size=32, shuffle=True, callbacks=[history])

# Získání historie trénování
loss_history = history.history['loss']
accuracy_history = history.history['accuracy']
