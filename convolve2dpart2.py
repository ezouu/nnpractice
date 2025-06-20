def make_synthetic_image():
    """Return a 100×100 list‑of‑lists; square in the middle is white (255)."""
    img = [[0.0 for _ in range(100)] for _ in range(100)]
    for i in range(30, 70):
        for j in range(30, 70):
            img[i][j] = 255.0
    return img

# Sobel‑x kernel as a plain nested list
sobel_x = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

def convolve2d_plain(image, kernel):
    """2‑D convolution using only Python built‑ins (no .shape, no np.pad)."""
    h, w = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])
    ph, pw = kh // 2, kw // 2        # half‑sizes for offsetting
    
    # output image filled with 0.0
    result = [[0.0 for _ in range(w)] for _ in range(h)]
    
    for i in range(h):
        for j in range(w):
            acc = 0.0
            # multiply‑and‑sum over the kernel neighbourhood
            for u in range(kh):
                for v in range(kw):
                    ii = i + u - ph
                    jj = j + v - pw
                    if 0 <= ii < h and 0 <= jj < w:   # treat out‑of‑bounds as 0
                        acc += kernel[u][v] * image[ii][jj]
            result[i][j] = acc
    return result

# Build image and apply kernel
img = make_synthetic_image()
grad_x = convolve2d_plain(img, sobel_x)

# Print EVERYTHING (100×100 each) as requested
print("=== Raw brightness matrix (img) ===")
for row in img:
    print(row)

print("\n=== Sobel‑x convolution result (grad_x) ===")
for row in grad_x:
    print(row)