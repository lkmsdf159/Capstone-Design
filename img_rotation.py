import cv2
import numpy as np
from PIL import Image
import re
import math
from paddleocr import PaddleOCR # main OCR dependencies
from matplotlib import pyplot as plt # plot images
import pandas as pd


def calculate_angle(img_path):

    ocr_model = PaddleOCR(lang='en')
    result = ocr_model.ocr(img_path)
    # Example usage:
    # Coordinates of the boxes in the format [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    try:
        box = result[0][0][0]
    except (TypeError, IndexError):
        print(f"No boxes in this picture: {img_path}")
        return None

    # Extract the coordinates of the box
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
    
    # Calculate the angles and lengths of each side with respect to the horizontal axis
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

    # Find the side with the longest length (assuming it should be horizontal)
    longest_side = max(sides, key=lambda s: s[1])
    angle_deg= longest_side[0]
    print("Angle of text of first box: ",angle_deg)


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
    # Calculate the center of the image
    center = (w / 2, h / 2)
    
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated

import os

img_path = 'cropped_imgs/bar/'
output_path = 'cropped_imgs/rotated_imgs/'
# List all files in the directory

valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
image_files = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f)) and f.lower().endswith(valid_extensions)]


for index, image_file in enumerate(image_files):
    image_path = os.path.join(img_path,image_file)
    final_image = calculate_angle(image_path)

    if final_image is not None:
        output_filename = f'{os.path.splitext(image_file)[0]}_{index}.jpg'
        cv2.imwrite(os.path.join(output_path,output_filename),final_image)
        print(f"Rotated Image saved to {output_path}")
    else:
        print(f"Skipping image {image_file} due to non detectable text box")
