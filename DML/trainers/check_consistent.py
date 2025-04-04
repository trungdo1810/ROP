import os

root_folder = '../datasets'
class_list = ['normal', 'pre-plus', 'plus']

def check_image_types(root_folder):
    print(f"Checking image types in folder: {root_folder}")
    for subdir, _, files in os.walk(root_folder):
        image_types = set()
        print(f"Checking folder: {subdir}")
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                image_types.add(file.split('.')[-1].lower())
        
        if len(image_types) > 1:
            print(f"Inconsistent image types found in folder: {subdir}")
            print(f"Image types: {image_types}")
        elif len(image_types) == 1:
            print(f"Consistent image type '{list(image_types)[0]}' found in folder: {subdir}")
        else:
            print(f"No images found in folder: {subdir}")

if __name__ == "__main__":
    print("Checking image types...")
    check_image_types(root_folder)