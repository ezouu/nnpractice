# ------------------------------------------------------------
# 1) Synthetic 100×100 white-square image
# ------------------------------------------------------------
def make_synthetic_image():
    img = [[0.0 for _ in range(100)] for _ in range(100)]
    for i in range(30, 70):          # rows 30-69
        for j in range(30, 70):      # cols 30-69
            img[i][j] = 255.0
    return img

def make_circle_image():
    img = [[0.0 for _ in range(100)] for _ in range(100)]
    cx, cy, r = 50, 50, 20                       # centre (50,50), radius 20
    for i in range(100):
        for j in range(100):
            if (i - cy) ** 2 + (j - cx) ** 2 <= r ** 2:
                img[i][j] = 255.0               # inside circle → white
    return img

# ------------------------------------------------------------
# 2) Plain 2-D convolution (no numpy helpers)
# ------------------------------------------------------------
def convolve2d_plain(image, kernel):
    h, w = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])
    ph, pw = kh // 2, kw // 2        # half sizes of the kernel

    # output image (same size) filled with zeros
    result = [[0.0 for _ in range(w)] for _ in range(h)]

    # slide kernel over every pixel
    for i in range(h):
        for j in range(w):
            acc = 0.0                       # accumulator
            for u in range(kh):
                for v in range(kw):
                    ii = i + u - ph         # image row under kernel cell
                    jj = j + v - pw         # image col under kernel cell
                    if 0 <= ii < h and 0 <= jj < w:   # zero-padding
                        acc += kernel[u][v] * image[ii][jj]
            result[i][j] = acc
    return result

# ------------------------------------------------------------
# 3) Sobel kernels
# ------------------------------------------------------------
sobel_x = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]
sobel_y = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
]

# ------------------------------------------------------------
# 4) Run everything
# ------------------------------------------------------------
img      = make_circle_image()
grad_x   = convolve2d_plain(img, sobel_x)   # vertical edges
grad_y   = convolve2d_plain(img, sobel_y)   # horizontal edges

# ------------------------------------------------------------
# 5) Print full matrices (will be big!)
# ------------------------------------------------------------
import sys
sys.setrecursionlimit(10000)                # allow huge prints

print("=== Raw brightness matrix (img) ===")
for row in img:
    print(row)

print("\n=== Sobel-x (vertical edges) grad_x ===")
for row in grad_x:
    print(row)

print("\n=== Sobel-y (horizontal edges) grad_y ===")
for row in grad_y:
    print(row)
