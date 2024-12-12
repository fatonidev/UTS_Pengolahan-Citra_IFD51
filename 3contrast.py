import imageio.v2 as imageio 
import numpy as np 
import matplotlib.pyplot as plt 

image = imageio.imread('citra.jpeg', mode='L') 

def adjust_contrast(img, factor):
    img = img / 255.0 
    img = np.clip(factor * (img - 0.5) + 0.5, 0, 1)
    return (img * 255).astype(np.uint8)

image_contrast = adjust_contrast(image, 1.5)

def histogram_equalization(img):
    img_flat = img.flatten()
    hist, bins = np.histogram(img_flat, bins=256, range=(0, 255), density=True)
    cdf = hist.cumsum()
    cdf_normalized = np.uint8(255 * cdf / cdf[-1])
    img_equalized = cdf_normalized[img_flat]
    img_equalized = img_equalized.reshape(img.shape)

    return img_equalized.astype(np.uint8)

image_enhanced = histogram_equalization(image)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Citra Asli')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Citra Setelah Contrast Level 1.5')
plt.imshow(image_contrast, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Citra Histogram Equalization')
plt.imshow(image_enhanced, cmap='gray')
plt.axis('off')

plt.show()
