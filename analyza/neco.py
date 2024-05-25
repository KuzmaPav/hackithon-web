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

def notch_filter(shape, notch_center, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2
    X, Y = np.ogrid[:rows, :cols]
    distance = np.sqrt((X - crow - notch_center[0])**2 + (Y - ccol - notch_center[1])**2)
    filter = np.ones(shape)
    filter[distance <= cutoff] = 0
    return filter

def apply_frequency_filter(image, filter_func, *args):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    filter = filter_func(image.shape, *args)
    filtered_dft = dft_shift * filter
    idft_shift = np.fft.ifftshift(filtered_dft)
    filtered_image = np.fft.ifft2(idft_shift)
    filtered_image = np.abs(filtered_image)
    return filtered_image

def evaluate_filters(image, original_image):
    # Apply the notch filter first
    notch_filtered_image = apply_frequency_filter(image, notch_filter, (30, 30), 10)

    # Define filters to apply after the notch filter
    filters = {
        'Median': cv2.medianBlur(notch_filtered_image.astype(np.uint8), 5),
        'Gaussian': cv2.GaussianBlur(notch_filtered_image.astype(np.uint8), (5, 5), 0),
        'Wavelet': denoise_wavelet(notch_filtered_image, method='BayesShrink', mode='soft'),
        'Total Variation': denoise_tv_chambolle(notch_filtered_image, weight=0.1)
    }
    
    # PSNR and SSIM
    scores = {}
    for name, filtered_image in filters.items():
        
        if name in ['Wavelet', 'Total Variation']:
            filtered_image = (filtered_image * 255).astype(np.uint8)
        psnr_value = psnr(original_image, filtered_image)
        ssim_value, _ = ssim(original_image, filtered_image, data_range=filtered_image.max() - filtered_image.min(), full=True)
        scores[name] = (psnr_value, ssim_value)
        print(f"{name} Filter - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")
    
    return notch_filtered_image, filters, scores

# dir
base_dir = './vzorky'

# Specify the target subfolder
target_subfolder = 'poisson_intensity_0.1'  # Replace with the desired subfolder name

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
    notch_filtered_image, filters, filter_scores = evaluate_filters(image, original_image)
    
    # plot notch filtered image
    plt.subplot(3, 3, 3)
    plt.title('Notch Filtered Image')
    plt.imshow(notch_filtered_image, cmap='gray')
    plt.axis('off')

    # print all filters with their metrics
    plot_index = 4
    for filter_name, filtered_image in filters.items():
        plt.subplot(3, 3, plot_index)
        plt.title(f'{filter_name} Filter\nPSNR: {filter_scores[filter_name][0]:.2f}, SSIM: {filter_scores[filter_name][1]:.4f}')
        plt.imshow(filtered_image if filter_name not in ['Wavelet', 'Total Variation'] else (filtered_image * 255).astype(np.uint8), cmap='gray')
        plt.axis('off')
        plot_index += 1
    
    plt.tight_layout()
    plt.show()
