import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Create a simple synthetic image (white square on black background)
img = np.zeros((100, 100), dtype=np.float32)
img[30:70, 30:70] = 255.0  # Add a white square

# Step 2: Define Sobel edge-detection kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

# Step 3: Convolution helper
def convolve2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

# Step 4: Apply filters to get gradients
grad_x = convolve2d(img, sobel_x)
grad_y = convolve2d(img, sobel_y)

# Step 5: Gradient magnitude & threshold
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
edges = (grad_mag > 100).astype(np.uint8) * 255

# Display original image (grayscale)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Synthetic Image')
plt.show()

# Display detected edges (grayscale)
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.title('Detected Edges')
plt.show()
