import cv2
import numpy as np
import os
from pathlib import Path


data_dir = Path('C:/Users/trung/Documents/AI/ROP/data')

def rename_files_in_folder(root_path):
    root = Path(root_path)

    for folder in root.iterdir():
        if folder.is_dir():
           prefix = folder.name.split("_")[0]
        
        for file in folder.iterdir():
            if file.is_file():
                new_name = f'{prefix}_{file.name}'
                file.rename(folder/new_name)
                print(f'Renamed in {folder.name}: {file.name} -> {new_name}')            

def convert_mask_to_yolo(img_folder, mask_folder, output_folder, class_id=0):
    # Ensure output folder exists
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Process each mask file in the folder
    for mask_file in Path(mask_folder).glob("*.png"):
        # Read the binary mask
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        # bin_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
       
        _, bin_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # bin_mask = cv2.resize(bin_mask, (640 ,480), interpolation=cv2.INTER_NEAREST)
        
        h, w = bin_mask.shape
        print(f"Image shape: height ({h}), width ({w})")

        # Find contours in the mask
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   

        # Prepare YOLO annotation data
        yolo_annotations = []
        for contour in contours:
            # print(type(contour))
            # print(contour.shape)
            # Flatten the contour points
            points = contour.squeeze()
            # print(type(points))
            # print(points.shape)

            if len(points.shape) != 2:  # Skip invalid contours
                continue
            
            # Normalize coordinates (x/w, y/h)
            normalized_points = [(round(float(x) / float(w), 6), round(float(y) / float(h), 6)) for x, y in points]
            flat_points = [coord for point in normalized_points for coord in point]
            
            # YOLO format: <class> <x1> <y1> <x2> <y2> ... <xn> <yn>
            annotation = [class_id] + flat_points
            yolo_annotations.append(annotation)
        
        # Write annotations to the .txt file
        output_file = output_folder / f"{mask_file.stem}.txt"
        with open(output_file, "w") as f:
            for annotation in yolo_annotations:
                f.write(" ".join(map(str, annotation)) + "\n")
        
        print(f"Converted: {mask_file.name} -> {output_file.name}")

        

output_folder = 'C:/Users/trung/Documents/AI/ROP/yolo_data/labels'
img_folder = 'C:/Users/trung/Documents/AI/ROP/data/images'
mask_folder = 'C:/Users/trung/Documents/AI/ROP/data/masks'
convert_mask_to_yolo(img_folder, mask_folder, output_folder)

# # Clean up
# while True:
#     if (cv2.waitKey(1) & 0xFF) == ord('q'):
#         break

# cv2.destroyAllWindows()