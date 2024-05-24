import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_wavelet, denoise_tv_chambolle

def get_random_image(base_dir):
    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir() and f.name != 'original']
    if not subfolders:
        raise ValueError("No valid subfolders found in the base directory.")
    
    random_folder = random.choice(subfolders)
    print(f"Selected folder: {random_folder}")

    images = [f for f in os.listdir(random_folder) if os.path.isfile(os.path.join(random_folder, f))]
    if not images:
        raise ValueError("No images found in the selected folder.")
    
    random_image = random.choice(images)
    image_path = os.path.join(random_folder, random_image)
    print(f"Selected image: {image_path}")
    return image_path

def evaluate_filters(image):
    # filters 
    filters = {
        'Median': cv2.medianBlur(image, 5),
        'Gaussian': cv2.GaussianBlur(image, (5, 5), 0),
        'Bilateral': cv2.bilateralFilter(image, 9, 75, 75),
        'Non-Local Means': cv2.fastNlMeansDenoising(image, None, 30, 7, 21),
        'Wavelet': denoise_wavelet(image, multichannel=False, method='BayesShrink', mode='soft'),
        'Total Variation': denoise_tv_chambolle(image, weight=0.1)
    }
    
    # PSNR and SSIM
    scores = {}
    for name, filtered_image in filters.items():
        
        if name in ['Wavelet', 'Total Variation']:
            filtered_image = (filtered_image * 255).astype(np.uint8)
        psnr_value = psnr(image, filtered_image)
        ssim_value, _ = ssim(image, filtered_image, full=True)
        scores[name] = (psnr_value, ssim_value)
        print(f"{name} Filter - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

    # best PSNR and SSIM
    best_filter = max(scores, key=lambda k: (scores[k][0], scores[k][1]))
    return best_filter, filters[best_filter], scores

# dir
base_dir = '/content/hackithon-web/vzorky'

# random image
random_image_path = get_random_image(base_dir)

# load
image = cv2.imread(random_image_path, cv2.IMREAD_GRAYSCALE)

# test
if image is None:
    print(f"Error: Image '{random_image_path}' not found.")
else:
    # original
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # the best filter ever exist
    best_filter_name, best_filtered_image, filter_scores = evaluate_filters(image)
    
    print(f"Best filter: {best_filter_name}")

    
    plt.subplot(2, 3, 2)
    plt.title(f'Best Filter: {best_filter_name}')
    plt.imshow(best_filtered_image if best_filter_name not in ['Wavelet', 'Total Variation'] else (best_filtered_image * 255).astype(np.uint8), cmap='gray')
    plt.axis('off')
    
    plt.show()