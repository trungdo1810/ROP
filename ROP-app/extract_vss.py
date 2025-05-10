import os
import torch
import numpy as np
from PIL import Image
from DML.models.CNN_model import EmbeddingModel
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define image preprocessing
def preprocess_image(image_path, image_size=500):
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    transformed = transform(image=image_np)
    return transformed["image"].unsqueeze(0)

# Load the model
model_path = r"C://Users//UCL//Desktop//Do//DML//checkpoints//CE_and_trip_EPSHN//fold1_best_auc0.9917_ep50.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingModel(num_classes=3, backbone_name="resnet50").to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Extract embeddings for reference images
reference_folder = "vss_landmark"
reference_embeddings = {}
for image_name in os.listdir(reference_folder):
    image_path = os.path.join(reference_folder, image_name)
    input_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        _, embedding = model(input_tensor)
    reference_embeddings[image_name] = embedding.cpu().numpy()

# Save reference embeddings
np.save("reference_embeddings.npy", reference_embeddings)
print("Reference embeddings saved.")