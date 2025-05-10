from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from pyngrok import ngrok
import torch
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from DML.models.CNN_model import EmbeddingModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set ngrok authtoken (replace with your actual token from ngrok dashboard)
authtoken = "2wsZwJrxgHlYzSdO97uudjFNP0G_7qbddYDzZBRq8BJxTwRg5"  # Thay bằng authtoken của bạn
ngrok.set_auth_token(authtoken)

# Load the model
model_path = r"C://Users//UCL//Desktop//Do//DML//checkpoints//CE_and_trip_EPSHN//fold1_best_auc0.9917_ep50.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingModel(num_classes=3, backbone_name="resnet50").to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load reference embeddings
reference_embeddings = np.load("reference_embeddings.npy", allow_pickle=True).item()

# Define image preprocessing
def preprocess_image_for_model(image_path, image_size=500):
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    transformed = transform(image=image_np)
    return image_np / 255.0, transformed["image"].unsqueeze(0).to(device)

def compute_similarity(embedding):
    """
    Compute cosine similarity between the input embedding and reference embeddings.
    Normalize the distances so their sum equals 1.
    """
    similarities = {}
    for name, ref_embedding in reference_embeddings.items():
        similarity = cosine_similarity(embedding.cpu().numpy(), ref_embedding)
        similarities[name] = similarity[0][0]

    # Debug: Print raw similarities
    print(f"Raw similarities: {similarities}")

    # Normalize similarities
    total_similarity = sum(similarities.values())
    if total_similarity == 0:
        normalized_similarities = {k: 0 for k in similarities.keys()}
    else:
        normalized_similarities = {k: v / total_similarity for k, v in similarities.items()}

    # Debug: Print normalized similarities
    print(f"Normalized similarities: {normalized_similarities}")

    return normalized_similarities

def generate_gradcam(image, threshold=0.5):
    """
    Generate Grad-CAM heatmap and predictions using the model.
    """
    # Preprocess the image
    img_np, input_tensor = preprocess_image_for_model(image)

    # Choose target layer
    target_layer = model.backbone.layer4[-1]

    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Get class index from model prediction
    with torch.no_grad():
        logits, embedding = model(input_tensor)
        class_idx = int(torch.argmax(logits))

    # Compute Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])[0]

    # Resize Grad-CAM heatmap to match the original image dimensions
    grayscale_cam_resized = cv2.resize(grayscale_cam, (img_np.shape[1], img_np.shape[0]))

    # Overlay Grad-CAM heatmap on the original image
    visualization = show_cam_on_image(img_np, grayscale_cam_resized, use_rgb=True)

    # Apply threshold
    heatmap_thresholded = np.copy(grayscale_cam_resized)
    heatmap_thresholded[heatmap_thresholded < threshold] = 0

    # Convert heatmaps to base64
    heatmap_img = Image.fromarray((grayscale_cam_resized * 255).astype(np.uint8))
    buffered = BytesIO()
    heatmap_img.save(buffered, format="PNG")
    heatmap_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    thresholded_img = Image.fromarray((heatmap_thresholded * 255).astype(np.uint8))
    buffered = BytesIO()
    thresholded_img.save(buffered, format="PNG")
    thresholded_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Compute similarity
    similarities = compute_similarity(embedding)
    print(f"Similarities: {similarities}")
    # Mock probabilities and diagnosis
    probabilities = torch.softmax(logits, dim=1).cpu().numpy().tolist()[0]
    diagnosis_labels = ["Normal", "Pre-Plus", "Plus"]
    diagnosis = diagnosis_labels[class_idx]

    return {
        "heatmap": heatmap_base64,
        "thresholded_heatmap": thresholded_base64,
        "probabilities": probabilities,
        "diagnosis": diagnosis,
        "similarities": similarities,
    }


@app.route("/test", methods=["POST"])
def test():
    if "image" in request.files:
        return jsonify({"message": "Image received"}), 200
    else:
        return jsonify({"error": "No image"}), 400


@app.route("/gradcam", methods=["POST"])
def gradcam():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file found in the request."}), 400

        image = request.files["image"]
        threshold = float(request.form.get("threshold", 0.5))

        # Process Grad-CAM
        result = generate_gradcam(image, threshold)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Start ngrok tunnel
    public_url = ngrok.connect(5000)
    print(f" * ngrok tunnel: {public_url}")

    # Run Flask app
    app.run(port=5000)
