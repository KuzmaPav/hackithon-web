import os
import numpy as np
from PIL import Image

def load_data(noisy_folder_path, clean_folder_path):
    # Count the number of noisy files
    num_files = len(os.listdir(noisy_folder_path))

    # Get image shape
    img_shape = np.array(Image.open(os.path.join(noisy_folder_path, os.listdir(noisy_folder_path)[0]))).shape

    # Initialize numpy arrays for noisy and clean data
    noisy_data = np.zeros((num_files, img_shape[0], img_shape[1]), dtype=np.uint8)
    clean_data = np.zeros((num_files, img_shape[0], img_shape[1]), dtype=np.uint8)

    # Index for tracking the position in numpy arrays
    index = 0

    # Iterate through each file in the noisy data folder
    for noisy_file in os.listdir(noisy_folder_path):
        # Construct the full file paths
        noisy_file_path = os.path.join(noisy_folder_path, noisy_file)
        clean_file_path = os.path.join(clean_folder_path, noisy_file)

        # Check if the corresponding clean data file exists
        if os.path.exists(clean_file_path):
            # Load the noisy and clean images
            noisy_image = Image.open(noisy_file_path)
            clean_image = Image.open(clean_file_path)

            # Convert images to numpy arrays
            noisy_array = np.array(noisy_image)
            clean_array = np.array(clean_image)

            # Assign the noisy and clean data arrays to numpy arrays
            noisy_data[index] = noisy_array
            clean_data[index] = clean_array

            # Increment the index
            index += 1

    # Resize the numpy arrays to remove empty slots
    noisy_data = noisy_data[:index]
    clean_data = clean_data[:index]

    return noisy_data, clean_data

