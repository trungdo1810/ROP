from torch.utils.data import Dataset
import os
import cv2
from PIL import Image
import numpy as np
import torch
import pandas as pd


class TripletDataset(Dataset):
    def __init__(self, root_dir, df, mode, transform=None):
        self.root = root_dir
        self.transform = transform
        self.mode = mode
        self.df = df
        self.img_path_list = df["path"].tolist()

        if "label" in df.columns:
            self.labels = df["label"].tolist()
        else:
            self.labels = None

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.img_path_list[idx])

        # Try OpenCV first (faster)
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("OpenCV couldn't read the image")
            # Convert BGR (OpenCV default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Fallback to PIL if OpenCV fails
        except:
            try:
                image = Image.open(image_path).convert("RGB")
                image = np.array(image)
            except Exception as e:
                raise Exception(
                    f"Failed to load image {image_path} with both OpenCV and PIL: {str(e)}"
                )

        # Apply transforms if provided
        if self.transform is not None:
            # Ensure image is in correct format for transforms
            transformed = self.transform(image=image)
            image = transformed["image"]

        if self.mode == "test":
            return image
        else:
            label = self.labels[idx]
            return image, torch.tensor(label).long()


if __name__ == "__main__":
    num_classes = 3
    root_dir = "../datasets/"
    csv_train_file = "train_data_with_folds.csv"
    class_list = ["normal", "preplus", "plus"]
    label_dict = {cls: i for i, cls in enumerate(class_list)}

    df = pd.read_csv(os.path.join(root_dir, csv_train_file))
    ds = TripletDataset(root_dir, df, "train", transform=None)
    # print(df.__len__())
