import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_wavelet, denoise_tv_chambolle

def get_random_image(base_dir, target_subfolder):
    target_folder_path = os.path.join(base_dir, target_subfolder)
    if not os.path.isdir(target_folder_path):
        raise ValueError(f"No valid subfolder found: {target_folder_path}")
    
    images = [f for f in os.listdir(target_folder_path) if os.path.isfile(os.path.join(target_folder_path, f))]
    if not images:
        raise ValueError(f"No images found in the selected folder: {target_folder_path}")
    
    random_image = random.choice(images)
    image_path = os.path.join(target_folder_path, random_image)
    print(f"Selected image: {image_path}")
    return image_path

def evaluate_filters(image, original_image):
    # filters 
    filters = {
        'Median': cv2.medianBlur(image, 5),
        'Gaussian': cv2.GaussianBlur(image, (5, 5), 0),
        'Bilateral': cv2.bilateralFilter(image, 9, 75, 75),
        'Non-Local Means': cv2.fastNlMeansDenoising(image, None, 30, 7, 21),
        'Wavelet': denoise_wavelet(image, method='BayesShrink', mode='soft'),
        'Total Variation': denoise_tv_chambolle(image, weight=0.1)
    }
    
    # PSNR and SSIM
    scores = {}
    for name, filtered_image in filters.items():
        
        if name in ['Wavelet', 'Total Variation']:
            filtered_image = (filtered_image * 255).astype(np.uint8)
        psnr_value = psnr(original_image, filtered_image)
        ssim_value, _ = ssim(original_image, filtered_image, full=True)
        scores[name] = (psnr_value, ssim_value)
        print(f"{name} Filter - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")
    
    return filters, scores

# dir
base_dir = './vzorky'

# Specify the target subfolder
target_subfolder = 'salt_and_pepper_speckle_intensity_0.03_0.3'

# random image
random_image_path = get_random_image(base_dir, target_subfolder)
random_image_name = os.path.basename(random_image_path)

# load original image
original_image_path = os.path.join(base_dir, 'original', random_image_name)
original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
if original_image is None:
    print(f"Error: Original image '{original_image_path}' not found.")
else:
    print(f"Loaded original image: {original_image_path}")

# load random image
image = cv2.imread(random_image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print(f"Error: Image '{random_image_path}' not found.")
else:
    print(f"Loaded random image: {random_image_path}")

# test
if image is not None and original_image is not None:
    # original
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.title('Noisy Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # apply filters and get scores
    filters, filter_scores = evaluate_filters(image, original_image)
    
    # print all filters with their metrics
    plot_index = 3
    for filter_name, filtered_image in filters.items():
        plt.subplot(3, 3, plot_index)
        plt.title(f'{filter_name} Filter\nPSNR: {filter_scores[filter_name][0]:.2f}, SSIM: {filter_scores[filter_name][1]:.4f}')
        plt.imshow(filtered_image if filter_name not in ['Wavelet', 'Total Variation'] else (filtered_image * 255).astype(np.uint8), cmap='gray')
        plt.axis('off')
        plot_index += 1
    
    plt.tight_layout()
    plt.show()
