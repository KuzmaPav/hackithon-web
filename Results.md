# Výsledky

```python
# Nastavení
    'Median': cv2.medianBlur(image, 5),
    'Gaussian': cv2.GaussianBlur(image, (5, 5), 0),
    'Bilateral': cv2.bilateralFilter(image, 9, 75, 75),
    'Non-Local Means': cv2.fastNlMeansDenoising(image, None, 30, 7, 21),
    'Wavelet': denoise_wavelet(image, method='BayesShrink', mode='soft'),
    'Total Variation': denoise_tv_chambolle(image, weight=0.1)
```

## poisson_intensity_0.1
### Vzorek 639
![alt text](./obrazky/639.png "639")
### Vzorek 655
![alt text](./obrazky/655.png "655")

## poisson_intensity_0.2
### Vzorek 753
![alt text](./obrazky/753.png "753")
### Vzorek 732
![alt text](./obrazky/732.png "732")

## poisson_intensity_0.3
### Vzorek 809
![alt text](./obrazky/809.png "809")
### Vzorek 843
![alt text](./obrazky/843.png "843")

## salt_and_pepper_speckle_intensity_0.01_0.1
### Vzorek 1125
![alt text](./obrazky/1125.png "1125")
### Vzorek 1169
![alt text](./obrazky/1169.png "1169")

## salt_and_pepper_speckle_intensity_0.03_0.3
### Vzorek 1057
![alt text](./obrazky/1057.png "1057")
### Vzorek 1007
![alt text](./obrazky/1007.png "1007")

## salt_and_pepper_speckle_intensity_0.05_0.2
### Vzorek 972
![alt text](./obrazky/972.png "972")
### Vzorek 949
![alt text](./obrazky/949.png "949")

## salt_pepper_intensity_0.01
### Vzorek 331
![alt text](./obrazky/331.png "331")
### Vzorek 358
![alt text](./obrazky/358.png "358")

## salt_pepper_intensity_0.03
### Vzorek 496
![alt text](./obrazky/496.png "496")
### Vzorek 465
![alt text](./obrazky/465.png "465")