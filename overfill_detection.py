from ultralytics import YOLO
import torch
from PIL import Image
import os
import pandas
import cv2

model = YOLO('./engineer_model//weights/best.pt')  

def load_image(img_path):
    return Image.open(img_path)

def crop(img_dir, bar_dir, overfill_dir, start_index):

    img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_path in img_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Processing image: {img_path}")

        results = model(img_path, save=True, imgsz=1024, conf=0.7)
        original_image = results[0].orig_img
    

        for i, (box, label) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls)):
            x1, y1, x2, y2 = box.tolist()
            cropped_object = original_image[int(y1):int(y2), int(x1):int(x2)]
                
            # Determine the category and corresponding directory
            category = model.names[int(label)]
            if category == 'bar':
                output_path = os.path.join(bar_dir, f"{img_name}_bar_{start_index}.jpg")
            elif category == 'overfill':
                output_path = os.path.join(overfill_dir, f"{img_name}_overfill_{start_index}.jpg")
            else:
                # If there are other categories, you can handle them here
                continue  # Skip unrecognized categories
                
            cv2.imwrite(output_path, cropped_object)
            start_index += 1




# Example usage
bar_dir = 'cropped_imgs/bar/'
overfill_dir = 'cropped_imgs/overfill/'
img_path = 'add_image/testttt/'
crop(img_path,bar_dir,overfill_dir, 0)
# detect_and_crop_by_class(model, img_path)

