import imageio.v2 as imageio  # Import imageio to read image files
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For displaying the image

# Membaca citra dengan mode 'L' untuk citra grayscale
image = imageio.imread('citra.jpeg', mode='L')  # Use 'mode=L' to read the image as grayscale

# Peningkatan kontras dengan faktor 1.5
def adjust_contrast(img, factor):
    img = img / 255.0  # Normalisasi ke 0-1
    img = np.clip(factor * (img - 0.5) + 0.5, 0, 1)  # Penyesuaian kontras
    return (img * 255).astype(np.uint8)

# Meningkatkan kontras citra dengan faktor 1.5
image_contrast = adjust_contrast(image, 1.5)

# Melakukan histogram equalization menggunakan numpy
def histogram_equalization(img):
    # Flatten the image array
    img_flat = img.flatten()

    # Compute the histogram
    hist, bins = np.histogram(img_flat, bins=256, range=(0, 255), density=True)

    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Normalize the CDF to fit the range [0, 255]
    cdf_normalized = np.uint8(255 * cdf / cdf[-1])

    # Map the image pixels to the equalized values
    img_equalized = cdf_normalized[img_flat]

    # Reshape the flattened image back to its original shape
    img_equalized = img_equalized.reshape(img.shape)

    return img_equalized.astype(np.uint8)

# Meningkatkan citra dengan histogram equalization
image_enhanced = histogram_equalization(image)

# Menampilkan hasil
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
