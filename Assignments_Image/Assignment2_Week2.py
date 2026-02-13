import os
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data

print("OpenCV:", cv2.__version__)

# helpers
def to_rgb(img_bgr_or_rgb):
    """Ensure image is RGB uint8."""
    if img_bgr_or_rgb is None:
        raise ValueError("Image is None. Check the path or loading step.")
    img = img_bgr_or_rgb
    if img.ndim == 2:
        return img
    # If it came from cv2.imread it's BGR; if from skimage it's RGB.
    # We detect by heuristic: assume cv2 format if loaded via cv2.imread.
    return img

def load_image(path=None):
    """Load an RGB image. If path is missing, fall back to a built-in sample."""
    if path and os.path.exists(path):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb
    # offline sample (RGB)
    return data.astronaut()

def show_side_by_side(img1, img2, title1="Image 1", title2="Image 2", cmap1=None, cmap2=None):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.imshow(img1, cmap=cmap1); plt.title(title1); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(img2, cmap=cmap2); plt.title(title2); plt.axis('off')
    plt.axis('off')
    plt.show()

def clip_uint8(x):
    """Clip and convert to uint8."""
    return np.clip(x, 0, 255).astype(np.uint8)


#  Load an image TODO:
img1 = "images/just.jpeg"

img = load_image(img1)
print("Image shape (H, W, C):", img.shape, "| dtype:", img.dtype)

plt.figure(figsize=(5,5))
plt.imshow(img)
plt.title("Loaded image (RGB)")
plt.axis("off")
plt.show()


# TODO (Task 2.1): print two pixel values
h, w = img.shape[:2]
print("Top-left pixel:", img[0, 0])
print("Center pixel:", img[h//2, w//2])
# Task 2.1 (Answer):
# An RGB image represents each pixel using three separate intensity values:
# Red (R), Green (G), and Blue (B). Each channel usually ranges from 0 to 255.
# A value of 0 means no contribution of that color, while 255 means full intensity.
# By combining these three numbers, different colors are produced.
# For example, [255, 0, 0] corresponds to pure red, and [255, 255, 255] represents white.



# TODO (Task 2.2): save and reload
output_path = "images/output_saved.png"

# OpenCV expects BGR when writing
bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_path, bgr)

reloaded_bgr = cv2.imread(output_path, cv2.IMREAD_COLOR)
reloaded_rgb = cv2.cvtColor(reloaded_bgr, cv2.COLOR_BGR2RGB)

print("Reloaded shape:", reloaded_rgb.shape, "| dtype:", reloaded_rgb.dtype)
show_side_by_side(img, reloaded_rgb, "Original", "Reloaded")
# After saving and reloading the image, its dimensions (height, width, channels)
# remain exactly the same as the original image.
# The data type is still uint8, which confirms that pixel precision was preserved.
# This demonstrates that OpenCV correctly stores and retrieves image data
# when proper RGB ↔ BGR conversion is applied.



# TODO (Task 3.1): grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
show_side_by_side(img, gray, "RGB", "Grayscale", cmap2="gray")
print("Gray shape:", gray.shape, "| dtype:", gray.dtype)
# The RGB image was converted into grayscale format,
# which reduces the image from three color channels to a single intensity channel.
# In grayscale images, each pixel stores only brightness information,
# not color. This simplifies many image processing tasks.


# TODO (Task 3.2): thresholding
th_manual = 128
_, binary_manual = cv2.threshold(gray, th_manual, 255, cv2.THRESH_BINARY)

_, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

show_side_by_side(binary_manual, binary_otsu, f"Manual (T={th_manual})", "Otsu", cmap1="gray", cmap2="gray")
print("Otsu threshold chosen by OpenCV:", _)
# Manual thresholding applies a fixed threshold value (128 in this case)
# to convert the grayscale image into a binary image.
# However, lighting conditions can vary between images.
# Otsu’s method automatically calculates an optimal threshold value
# based on the image histogram, often producing a more balanced segmentation result.



# TODO (Task 4.1): set ROI coordinates
h, w = img.shape[:2]
x1, y1 = int(0.25*w), int(0.25*h)
x2, y2 = int(0.75*w), int(0.75*h)

roi = img[y1:y2, x1:x2].copy()
show_side_by_side(img, roi, "Original", f"ROI x[{x1}:{x2}] y[{y1}:{y2}]")
print("ROI shape:", roi.shape)
# A Region of Interest (ROI) was extracted using NumPy slicing.
# The format img[y1:y2, x1:x2] selects a rectangular portion of the image.
# In this case, the central area was selected, which often contains
# the main subject of the image.



# Task 4.2 (Answer):
# Images in Python are stored as NumPy arrays using row-major order.
# This means indexing follows the format img[row, column] or img[y, x].
# The first index corresponds to vertical position (height),
# and the second index corresponds to horizontal position (width).



# TODO (Task 5.1): HSV conversion and channel display
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

plt.figure(figsize=(12,3))
for i, (ch, name) in enumerate([(H,"H"), (S,"S"), (V,"V")], start=1):
    plt.subplot(1,3,i); plt.imshow(ch, cmap="gray"); plt.title(name); plt.axis("off")
plt.show()
# The RGB image was converted to the HSV color space, which separates color information (Hue), color intensity (Saturation), and brightness (Value).
# Displaying the H, S, and V channels individually helps visualize how color, purity, and brightness are represented independently in an image.


# 6.1 Provided example (edit the constant and re-run)
sub_val = 100  # TODO: try 30, 100, 150
img_sub = cv2.subtract(img, sub_val)

show_side_by_side(img, img_sub, "Original", f"Subtracted (-{sub_val})")
# Subtracting a constant value from the image decreases overall brightness.
# As the subtraction value increases, the image becomes darker.
# OpenCV ensures that pixel values do not go below 0,
# which prevents underflow.



# TODO (Task 6.2): subtract from Red channel only
sub_val_r = 80

img_red_only = img.copy()
# RGB: channel 0=R, 1=G, 2=B
img_red_only[:,:,0] = cv2.subtract(img_red_only[:,:,0], sub_val_r)

show_side_by_side(img, img_red_only, "Original", f"Red channel subtracted (-{sub_val_r})")
# Only the red channel was modified while green and blue remained unchanged.
# Reducing values in the red channel decreases warm tones in the image,
# giving it a cooler appearance.



#Task 6.3
add_val = 100  # TODO: try 30, 100, 150
img_add = cv2.add(img, add_val)

show_side_by_side(img, img_add, "Original", f"Added (+{add_val})")
# Adding a constant value increases pixel intensities,
# which makes the image brighter.
# OpenCV automatically clips values at 255 to prevent overflow.



# Task 6.4
# TODO (Task 6.4): add to Red channel only
add_val_r = 80

img_red_only_add = img.copy()
img_red_only_add[:,:,0] = cv2.add(img_red_only_add[:,:,0], add_val_r)

show_side_by_side(img, img_red_only_add, "Original", f"Red channel added (+{add_val_r})")
# Increasing only the red channel strengthens warm color tones.
# Since other channels are unchanged, the image gains a reddish tint.



# 6.5 TODO: try factors like 0.5, 1.2, 2.0
factor = 2.0

img_f = img.astype(np.float32)
img_mul = clip_uint8(img_f * factor)

show_side_by_side(img, img_mul, "Original", f"Multiplied (×{factor})")
# Multiplying pixel values by a factor greater than 1
# increases brightness and contrast, but may cause saturation.
# A factor less than 1 reduces brightness and lowers contrast.


# 6.6 TODO: try divisors like 2.0, 3.0, 0.5
divisor = 2.0

img_f = img.astype(np.float32)
img_div = clip_uint8(img_f / divisor)

show_side_by_side(img, img_div, "Original", f"Divided (÷{divisor})")
# Dividing pixel values by a number greater than 1 decreases brightness.
# If the divisor is less than 1, brightness increases.
# Clipping ensures values remain within the valid 0–255 range.



# Report:
# In this assignment, I loaded an image using OpenCV, checked its size
# and some pixel values, saved and reloaded the image to make sure it was not changed,
# then converted it to grayscale and HSV color spaces, applied manual and Otsu thresholding,
# selected a region of interest from the image, and finally changed the brightness and color channels by adding,
# subtracting, and multiplying pixel values to see how these operations affect the image.