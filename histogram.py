import imageio
import numpy as np
import matplotlib.pyplot as plt

# Membaca citra dengan mode 'L' untuk citra grayscale
image = imageio.imread('citra.jpeg', mode='L')

# Histogram Equalization
def histogram_equalization(img):
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()  # Cumulative Distribution Function
    cdf_normalized = cdf * 255 / cdf[-1]  # Normalisasi
    img_equalized = np.interp(img.flatten(), bins[:-1], cdf_normalized).reshape(img.shape)
    return img_equalized

# Meningkatkan kontras citra
image_enhanced = histogram_equalization(image)

# Menampilkan hasil
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Citra Asli')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Citra Setelah Histogram Equalization')
plt.imshow(image_enhanced, cmap='gray')
plt.axis('off')
plt.show()
