import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load image paths and labels
        for label, folder in enumerate(["crop_normal", "crop_plus", "crop_preplus"]):
            folder_path = os.path.join(data_dir, folder)
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg"):
                    self.image_paths.append(os.path.join(folder_path, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_data_loaders(data_dir, batch_size, train_ratio=0.8, shuffle=True):
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((1200, 1200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create dataset
    dataset = CustomDataset(data_dir, transform=transform)

    # Split dataset into train and validation sets
    dataset_size = len(dataset)
    # print(dataset_size)

    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    n_workers = os.cpu_count()
    print("num_workers = ", n_workers)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    return train_loader, val_loader


if __name__ == "__main__":
    data_dir = "C:\\Users\\UCL\\Desktop\\Do\\DML\\datasets"
    batch_size = 32
    epochs = 50

    # Create data loaders
    train_loader, val_loader = create_data_loaders(data_dir, batch_size)
