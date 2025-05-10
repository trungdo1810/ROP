import sys
import os
# Get the root directory: one level up from "trainers"
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(ROOT_DIR)

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import os

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.CNN_model import EmbeddingModel

def load_image(image_path, image_size=500):
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    transformed = transform(image=image_np)
    return image_np / 255.0, transformed["image"].unsqueeze(0)


def main(model_path, image_path, output_path):
    # Load model
    model = EmbeddingModel(num_classes=3, backbone_name="resnet50")
    model.load_state_dict(torch.load(model_path))
    # model = torch.load(model_path, map_location='cpu')
    model.eval()

    # Load and preprocess image
    img_np, input_tensor = load_image(image_path)

    # Choose target layer
    # print(model.backbone.layer4[-1])
    target_layer = model.backbone.layer4[-1]

    # Initialize GradCAM
    cam = GradCAM(
        model=model, target_layers=[target_layer], 
    )

    # Get class index from model prediction
    with torch.no_grad():
        logits, _ = model(input_tensor)
        class_idx = int(torch.argmax(logits))

    # Compute GradCAM
    grayscale_cam = cam(
        input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)]
    )[0]

    # Resize Grad-CAM heatmap to match the original image dimensions
    grayscale_cam_resized = cv2.resize(grayscale_cam, (img_np.shape[1], img_np.shape[0]))

    # Overlay and save
    visualization = show_cam_on_image(img_np, grayscale_cam_resized, use_rgb=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    # Show original image, heatmap, and overlay
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis("off")

    # Grad-CAM heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(grayscale_cam_resized, cmap="jet")
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(visualization)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model_path = r"C://Users//UCL//Desktop//Do//DML//checkpoints//CE_and_trip_EPSHN//fold1_best_auc0.9917_ep50.pth"
    image_path = r"C://Users//UCL//Desktop//Do//DML//datasets//plus//1d8dcdd1-bc8b-4c1c-be19-fcd4935aad2e.5.jpg"

    parser = argparse.ArgumentParser(description="GradCAM Visualization for ResNet50")
    # parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pth)")
    # parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/gradcam_result.jpg",
        help="Path to save result image",
    )
    args = parser.parse_args()

    main(model_path, image_path, args.output)
