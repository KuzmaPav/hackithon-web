import os
import numpy as np
from PIL import Image

def load_data(noisy_folder_path, clean_folder_path):
    # Get list of file names
    noisy_files = os.listdir(noisy_folder_path)

    # Filter out files that don't have corresponding clean files
    valid_files = [file for file in noisy_files if os.path.exists(os.path.join(clean_folder_path, file))]

    # Get number of valid files
    num_files = len(valid_files)

    # Get image shape
    img_shape = np.array(Image.open(os.path.join(noisy_folder_path, valid_files[0]))).shape

    # Initialize numpy array for data
    data = np.zeros((num_files, 2, img_shape[0], img_shape[1]), dtype=np.uint8)

    # Iterate through each valid file
    for i, file in enumerate(valid_files):
        # Construct the full file paths
        noisy_file_path = os.path.join(noisy_folder_path, file)
        clean_file_path = os.path.join(clean_folder_path, file)

        # Load the noisy and clean images
        noisy_image = Image.open(noisy_file_path)
        clean_image = Image.open(clean_file_path)

        # Convert images to numpy arrays
        noisy_array = np.array(noisy_image)
        clean_array = np.array(clean_image)

        # Assign the noisy and clean data arrays to numpy array
        data[i][0] = noisy_array
        data[i][1] = clean_array

    return data

# Example usage
noise_data_path = r"vzorky\poisson_intensity_0.1"
clean_data_path = r"vzorky\original"
data = load_data(noise_data_path, clean_data_path)
print(data.shape)  # Should print (num_files, 2, img_height, img_width)
