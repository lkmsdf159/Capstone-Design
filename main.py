from ultralytics import YOLO
from PIL import Image
import os
import pandas as pd
import math
from paddleocr import PaddleOCR, draw_ocr 
from matplotlib import pyplot as plt 
import cv2
import re
import numpy as np
from scipy.spatial import distance
import warnings
warnings.filterwarnings("ignore")

import logging
# Set the logging level to ERROR to suppress warnings and debug messages
logging.getLogger('ppocr').setLevel(logging.ERROR)

# Initialize YOLO models for detection and segmentation
model = YOLO('./engineer_model//weights/best.pt')  
ocr_model = PaddleOCR(lang='en')
seg_model = YOLO('seg_dir/best.pt')

# Function to load an image from a given path
def load_image(img_path):
    return Image.open(img_path)

# Function to detect bars and overfills in an image, crop them, and save the cropped images
def detect_bar_overfill(img_path, bar_dir, overfill_dir, start_index):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"Processing image: {img_path}")

    # Perform object detection on the image
    results = model(img_path, save=True, imgsz=1024, conf=0.8)
    original_image = results[0].orig_img

    cropped_images = {'bar': [], 'overfill': []}

    # Loop through detected objects and save the cropped images
    for i, (box, label) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls)):
        x1, y1, x2, y2 = box.tolist()
        cropped_object = original_image[int(y1):int(y2), int(x1):int(x2)]
            
        category = model.names[int(label)]
        if category == 'bar':
            output_path = os.path.join(bar_dir, f"{img_name}_bar_{start_index}.jpg")
            cropped_images['bar'].append(output_path)
        elif category == 'overfill':
            output_path = os.path.join(overfill_dir, f"{img_name}_overfill_{start_index}.jpg")
            cropped_images['overfill'].append(output_path)
        else:
            continue  
            
        cv2.imwrite(output_path, cropped_object)
        start_index += 1

    return cropped_images

# Function to measure the area and length of overfill in an image
def overfill_measurement(image_path):
    # Read the image
    img = cv2.imread(image_path)
    area_length = []
    H, W, _ = img.shape

    # Perform segmentation to get masks
    results = seg_model.predict(image_path, save=True)

    # Variable to store the first mask
    first_mask = None

    # Retrieve only the first mask
    for i, result in enumerate(results):
        if result.masks is not None and result.masks.data is not None:
            for j, mask in enumerate(result.masks.data):
                first_mask = mask.cpu().numpy() * 255
                first_mask = cv2.resize(first_mask, (W, H)).astype(np.uint8)
                break
        if first_mask is not None:
            break

    if first_mask is not None:
        # Process the first mask
        binary_mask = first_mask
        _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
        scale_cm_per_pixel = 0.05  # scale in cm per pixel
        scale_m_per_pixel = scale_cm_per_pixel / 100.0

        # Calculate the area of the white region in pixels
        white_area_pixels = np.sum(binary_mask == 255)
        white_area_m2 = white_area_pixels * (scale_m_per_pixel ** 2)
        area_length.append(white_area_m2)

        # Find the coordinates of all white pixels
        white_pixels = np.column_stack(np.where(binary_mask == 255))

        if len(white_pixels) > 1:
            max_distance_pixels = np.max(distance.cdist(white_pixels, white_pixels, 'euclidean'))
            max_distance_m = max_distance_pixels * scale_m_per_pixel
        else:
            print("Not enough white pixels to calculate the maximum distance.")
        area_length.append(max_distance_m)
        mask_filename = f'{os.path.splitext(os.path.basename(image_path))[0]}_mask.png'
        mask_path = os.path.join(masks_dir, mask_filename)
        cv2.imwrite(mask_path, first_mask)

    return area_length

# Function to calculate the angle of text and rotate the image to align the text horizontally
def calculate_angle_rotate(img_path, output_path):
    result = ocr_model.ocr(img_path)
    try:
        box = result[0][0][0]
    except (TypeError, IndexError):
        print(f"No boxes in this picture:")
        return None

    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
    
    sides = []
    angles = []

    for (x_start, y_start), (x_end, y_end) in [(box[0], box[1]), (box[1], box[2]), (box[2], box[3]), (box[3], box[0])]:
        dx = x_end - x_start
        dy = y_end - y_start
        length = math.sqrt(dx**2 + dy**2)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        angle_sin = math.asin(dy/length)
        angle_sing = math.degrees(angle_sin)
        sides.append((angle_sing, length))
        angles.append(angle_sing)

    longest_side = max(sides, key=lambda s: s[1])
    angle_deg= longest_side[0]
    print("Angle of text of first box: ",angle_deg)

    # Determine the rotation angle based on the calculated angle
    if angle_deg > 100 and angle_deg<=130:
        rotation_angle = -angle_deg
    elif angle_deg > 130:
        rotation_angle = 0
    elif angle_deg > 0:
        rotation_angle = angle_deg
    elif angle_deg < 0 and angle_deg > -10:
        rotation_angle =  angle_deg - 180
    elif angle_deg < 0 and angle_deg>-180:
        rotation_angle = -angle_deg
    elif angle_deg<0:
        rotation_angle= angle_deg
    else:
        rotation_angle = angle_deg + 360

    print(rotation_angle)
   
    image= cv2.imread(img_path)

    if image is None:
        print(f"Failed to read image: {img_path}")
        return None

    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    cropped_path = os.path.join(output_path, f"{img_name}_cropped_{0}.jpg")
    cv2.imwrite(cropped_path, rotated)
    
    return rotated

# Function to perform OCR on a rotated image and extract text
def rotation_ocr(image_path):
    final_image = calculate_angle_rotate(image_path, rotated_output_path)
    result = ocr_model.ocr(final_image)
    
    text_ocr = []
    for i in range(len(result[0])):
        if len(result[0][i][1][0])>2:
            text_ocr.append(result[0][i][1][0])
    return text_ocr

# Function to add extracted data to a pandas DataFrame
def add_data_to_dataframe(df, text_ocr, overfill_path, filename):
    print(text_ocr)
    heat_number = text_ocr[0]
    id_number= text_ocr[1]
    if not overfill_path:
        overfill_occurance = "N"
        overfill_area = 0
        overfill_length = 0
        overfill_severity = "A"
    else:
        print(overfill_path[0])
        overfill_occurance = "Y"
        area_length = overfill_measurement(overfill_path[0])
        overfill_area = area_length[0]
        overfill_length = area_length[1]
        overfill_severity = "C"

    if heat_number[0] == '4' or heat_number[0] == '2': 
        heat_number = 'A' + heat_number[1:]
    elif heat_number[0] == '3' or heat_number[0] == '6' or heat_number[0] == '8':
        heat_number = 'B' + heat_number[1:]

    if len(id_number)>4:
        result = re.findall(r'\d', id_number)
        if(len(result)>=4):
            id_number = ''.join(result[:4])
        else:
            id_number = "N/A"
            overfill_severity = "D"
    elif len(id_number)<=3:
        overfill_severity = "D"
        id_number = "N/A"

    df.loc[len(df), 'HEAT'] = heat_number
    df.loc[len(df) - 1, 'ID'] = id_number
    df.loc[len(df) - 1, 'Overfill(Y,N)'] = overfill_occurance
    df.loc[len(df) - 1, 'Area'] = overfill_area
    df.loc[len(df) - 1, 'Length'] = overfill_length
    df.loc[len(df) - 1, 'Severity'] = overfill_severity
    df.loc[len(df) - 1, 'FileName'] = filename

# Set directories for saving images and masks
rotated_output_path = 'cropped_imgs/rotated_imgs/'
bar_dir = 'cropped_imgs/bar/'
overfill_dir = 'cropped_imgs/overfill/'
masks_dir = 'cropped_imgs/masks'

# Directory containing the input images
img_path = 'demo_images/'

# Create an empty DataFrame to store results
df = pd.DataFrame(columns=['HEAT', 'ID', 'Overfill(Y,N)', 'Area', 'Length', 'Severity','FileName'])

# List all valid image files in the input directory
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
image_files = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f)) and f.lower().endswith(valid_extensions)]

# Process each image file
for index, image_file in enumerate(image_files):
    image_path = os.path.join(img_path,image_file)
    cropped_images = detect_bar_overfill(image_path, bar_dir, overfill_dir, 0)
    bar_image =  cropped_images['bar']
    overfill_image = cropped_images['overfill']
    text_ocr = rotation_ocr(bar_image[0])
    add_data_to_dataframe(df, text_ocr, overfill_image, image_file)

# Save the final DataFrame to a CSV file
print(df)
df.to_csv('final_report.csv')
