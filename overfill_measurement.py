import cv2
import numpy as np
from scipy.spatial import distance

mask_path = 'seg_dir/masks/Overfill_20_overfill_1_mask_0_0.png'
binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Ensure the binary mask is binary
_, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

# Scale (in cm per pixel)
scale_cm_per_pixel = 0.05  # your provided scale

# Convert scale to meters per pixel
scale_m_per_pixel = scale_cm_per_pixel / 100.0

# Calculate the area of the white region in pixels
white_area_pixels = np.sum(binary_mask == 255)

# Calculate the real area in square meters
white_area_m2 = white_area_pixels * (scale_m_per_pixel ** 2)
print(f"Area of the white region: {white_area_m2} square meters")

# Find the coordinates of all white pixels
white_pixels = np.column_stack(np.where(binary_mask == 255))

# Calculate the maximum distance between any two white pixels
if len(white_pixels) > 1:
    max_distance_pixels = np.max(distance.cdist(white_pixels, white_pixels, 'euclidean'))
    max_distance_m = max_distance_pixels * scale_m_per_pixel
    print(f"Maximum length from one boundary to the other: {max_distance_m} meters")
else:
    print("Not enough white pixels to calculate the maximum distance.")