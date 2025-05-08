from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from pyngrok import ngrok

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set ngrok authtoken (replace with your actual token from ngrok dashboard)
authtoken = "YOUR_NGROK_AUTHTOKEN"  # Thay bằng authtoken của bạn
ngrok.set_auth_token(authtoken)

def generate_gradcam(image, threshold=0.5):
    """
    Placeholder for Grad-CAM generation
    Replace with actual model inference and Grad-CAM computation
    """
    img = Image.open(image).convert("RGB")
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # Create a mock heatmap (placeholder)
    heatmap = np.zeros((height, width))
    for _ in range(5):
        center_y = np.random.randint(height//4, 3*height//4)
        center_x = np.random.randint(width//4, 3*width//4)
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        max_dist = np.sqrt(height**2 + width**2)
        intensity = np.maximum(0, 1 - dist / (max_dist/3))
        heatmap += intensity
    
    # Normalize to [0, 1]
    heatmap = heatmap / heatmap.max()
    
    # Apply threshold
    heatmap_thresholded = np.copy(heatmap)
    heatmap_thresholded[heatmap_thresholded < threshold] = 0
    
    # Convert heatmaps to base64
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    buffered = BytesIO()
    heatmap_img.save(buffered, format="PNG")
    heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    thresholded_img = Image.fromarray((heatmap_thresholded * 255).astype(np.uint8))
    buffered = BytesIO()
    thresholded_img.save(buffered, format="PNG")
    thresholded_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Mock probabilities and diagnosis
    probabilities = np.random.dirichlet(np.ones(9), size=1)[0].tolist()
    diagnosis_labels = ["Normal", "Pre-Plus", "Plus"]
    diagnosis = np.random.choice(diagnosis_labels, p=[0.5, 0.3, 0.2])
    
    return {
        "heatmap": heatmap_base64,
        "thresholded_heatmap": thresholded_base64,
        "probabilities": probabilities,
        "diagnosis": diagnosis
    }

@app.route('/test', methods=['POST'])
def test():
    if 'image' in request.files:
        return jsonify({"message": "Image received"}), 200
    else:
        return jsonify({"error": "No image"}), 400

@app.route('/gradcam', methods=['POST'])
def gradcam():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file found in the request."}), 400
        
        image = request.files['image']
        threshold = float(request.form.get('threshold', 0.5))
        
        # Process Grad-CAM
        result = generate_gradcam(image, threshold)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start ngrok tunnel
    public_url = ngrok.connect(5000)
    print(f" * ngrok tunnel: {public_url}")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)