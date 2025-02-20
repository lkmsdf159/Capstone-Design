from ultralytics import YOLO
import cv2
import os
import torch
import numpy as np
from scipy.spatial import distance

image_path = 'cropped_imgs/overfill/Overfill_1_overfill_1.jpg'

model_path = 'seg_dir/best.pt'

# Load the model
model = YOLO(model_path)

# Ensure output directory exists
output_dir = './cropped_imgs/masks'
os.makedirs(output_dir, exist_ok=True)

# Read the image
img = cv2.imread(image_path)

H, W, _ = img.shape

# Predict and save results
results = model.predict(image_path, save=True, device='cpu')  # Use appropriate device

# Variable to store the first mask
first_mask = None

# Retrieve only the first mask
for i, result in enumerate(results):
    if result.masks is not None and result.masks.data is not None:
        for j, mask in enumerate(result.masks.data):
            first_mask = mask.cpu().numpy() * 255  # Move tensor to CPU before converting to numpy
            first_mask = cv2.resize(first_mask, (W, H)).astype(np.uint8)
            break  # Exit after getting the first mask
    if first_mask is not None:
        break  # Exit the outer loop as well

if first_mask is not None:
    # Process the first mask
    binary_mask = first_mask
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

    # Find the coordinates of all white pixels
    white_pixels = np.column_stack(np.where(binary_mask == 255))

    # Calculate the maximum distance between any two white pixels
    if len(white_pixels) > 1:
        max_distance_pixels = np.max(distance.cdist(white_pixels, white_pixels, 'euclidean'))
        max_distance_m = max_distance_pixels * scale_m_per_pixel
    else:
        print("Not enough white pixels to calculate the maximum distance.")
    
    # Save the first mask
    mask_filename = f'{os.path.splitext(os.path.basename(image_path))[0]}_mask.png'
    mask_path = os.path.join(output_dir, mask_filename)
    print(white_area_m2)
    print(max_distance_m)
    cv2.imwrite(mask_path, first_mask)



