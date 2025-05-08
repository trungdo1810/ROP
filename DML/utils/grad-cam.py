
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

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Resize((224, 224))
    resized_image = transform(image)
    img_np = np.array(resized_image) / 255.0  # [0, 1] normalized
    return img_np, preprocess_image(img_np, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

def main(model_path, image_path, output_path):
    # Load model
    model = model.load_state_dict(torch.load(model_path))
    # model = torch.load(model_path, map_location='cpu')
    model.eval()

    # Load and preprocess image
    img_np, input_tensor = load_image(image_path)

    # Choose target layer
    target_layer = model.layer4[-1]

    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

    # Get class index from model prediction
    with torch.no_grad():
        output = model(input_tensor)
        class_idx = int(torch.argmax(output))

    # Compute GradCAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])[0]

    # Overlay and save
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    # Show image
    plt.imshow(visualization)
    plt.title(f"Grad-CAM for class {class_idx}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    model_path = r'C:\\Users\\UCL\\Desktop\\Do\\DML\\checkpoints\\CE_and_trip_EPSHN\\fold1_best_auc0.9917_ep50.pth'
    image_path = r'C:\\Users\\UCL\\Desktop\\Do\\DML\\datasets\\plus\\1d8dcdd1-bc8b-4c1c-be19-fcd4935aad2e.1.jpg'

    parser = argparse.ArgumentParser(description="GradCAM Visualization for ResNet50")
    # parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pth)")
    # parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="outputs/gradcam_result.jpg", help="Path to save result image")
    args = parser.parse_args()

    main(model_path, image_path, args.output)
