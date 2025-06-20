import numpy as np

def make_synthetic_image():
    """
    Returns a 100×100 float32 array.
    Pixels inside the square [30:70, 30:70] are 255 (white);
    everything else is 0 (black).
    """
    img = np.zeros((100, 100), dtype=np.float32)
    img[30:70, 30:70] = 255.0
    return img

# ---- create the image ----
img = make_synthetic_image()

# ---- have a look (optional) ----
np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # show every value
print("Raw brightness values:\n", img)
