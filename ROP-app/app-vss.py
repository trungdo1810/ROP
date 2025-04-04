import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io

# For GradCAM visualization (add your actual implementation)
def generate_gradcam(image, threshold=0.5):
    """
    Placeholder for GradCAM generation
    In production, replace with actual model inference and GradCAM generation
    """
    # Create a mock heatmap that would be replaced by actual GradCAM output
    img = Image.open(image)
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # Create a simple heatmap (this is just a placeholder)
    heatmap = np.zeros((height, width))
    
    # Create some random "hot" areas for demonstration
    for _ in range(5):
        center_y = np.random.randint(height//4, 3*height//4)
        center_x = np.random.randint(width//4, 3*width//4)
        
        # Create a gradient from center
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        max_dist = np.sqrt(height**2 + width**2)
        
        # Create a circular pattern
        intensity = np.maximum(0, 1 - dist / (max_dist/3))
        heatmap += intensity
    
    # Normalize to [0, 1]
    heatmap = heatmap / heatmap.max()
    
    # Apply threshold
    heatmap_thresholded = np.copy(heatmap)
    heatmap_thresholded[heatmap_thresholded < threshold] = 0
    
    return heatmap, heatmap_thresholded

def generate_heatmap(probabilities):
    fig, ax = plt.subplots(figsize=(10, 0.5))
    sns.heatmap(
        [probabilities],
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        cbar=False,
        xticklabels=range(1, 10),
        yticklabels=["VSS"],
    )
    return fig

def overlay_heatmap(image, heatmap):
    """Create an overlay of the original image with the heatmap"""
    # Convert PIL Image to numpy array if needed
    if not isinstance(image, np.ndarray):
        img = Image.open(image)
        img_array = np.array(img)
    else:
        img_array = image
    
    # Resize heatmap if dimensions don't match
    if img_array.shape[:2] != heatmap.shape:
        heatmap_resized = np.zeros(img_array.shape[:2])
        h_ratio = img_array.shape[0] / heatmap.shape[0]
        w_ratio = img_array.shape[1] / heatmap.shape[1]
        
        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                h_idx = min(int(i / h_ratio), heatmap.shape[0] - 1)
                w_idx = min(int(j / w_ratio), heatmap.shape[1] - 1)
                heatmap_resized[i, j] = heatmap[h_idx, w_idx]
        heatmap = heatmap_resized
    
    # Create a colored heatmap (red for high attention)
    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    
    # Blend the images
    alpha = 0.4  # Transparency factor
    overlay = (1 - alpha) * img_array / 255.0 + alpha * heatmap_colored
    
    # Ensure values are in [0, 1]
    overlay = np.clip(overlay, 0, 1)
    
    return overlay

# Streamlit UI
def main():
    st.set_page_config(layout="wide")
    
    # Make the app more responsive
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 95%;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("Vascular Severity Assessment")

    # Initialize session state
    if "image_index" not in st.session_state:
        st.session_state.image_index = 0
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.5
    if "show_gradcam" not in st.session_state:
        st.session_state.show_gradcam = False

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Fundus Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True
    )

    if uploaded_files:
        num_images = len(uploaded_files)

        # Navigation buttons in a smaller container
        nav_col1, nav_col2, nav_col3 = st.columns([1, 6, 1])
        with nav_col1:
            if st.button("⬅️ Prev") and st.session_state.image_index > 0:
                st.session_state.image_index -= 1
        with nav_col3:
            if st.button("Next ➡️") and st.session_state.image_index < num_images - 1:
                st.session_state.image_index += 1

        # Get current image and filename
        current_image = uploaded_files[st.session_state.image_index]
        image_name = current_image.name
        
        # Create a buffer to allow the image to be read multiple times
        image_bytes = current_image.getvalue()
        current_image_buf = io.BytesIO(image_bytes)

        # Image display section
        st.subheader("Fundus Image Analysis")
        
        # Toggle GradCAM visualization
        st.session_state.show_gradcam = st.checkbox("Show GradCAM Visualization", value=st.session_state.show_gradcam)
        
        # Image display columns
        if st.session_state.show_gradcam:
            # Display original image and GradCAM side by side
            img_col1, img_col2 = st.columns([1, 1])
            
            with img_col1:
                st.markdown("### Original Image")
                st.image(
                    current_image_buf,
                    caption=f"{image_name} (Image {st.session_state.image_index + 1} of {num_images})",
                    use_container_width=True,
                )
            
            with img_col2:
                st.markdown("### GradCAM Visualization")
                
                # Watershed threshold slider
                threshold = st.slider(
                    "Attention Threshold", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=st.session_state.threshold,
                    step=0.05
                )
                st.session_state.threshold = threshold
                
                # Generate GradCAM visualization
                heatmap, thresholded_heatmap = generate_gradcam(io.BytesIO(image_bytes), threshold)
                
                # Create tabs for different visualizations
                gradcam_tab1, gradcam_tab2 = st.tabs(["Heatmap Overlay", "Raw Heatmap"])
                
                with gradcam_tab1:
                    # Display overlay image
                    overlay = overlay_heatmap(io.BytesIO(image_bytes), thresholded_heatmap)
                    st.image(
                        overlay,
                        caption="GradCAM Overlay",
                        use_container_width=True,
                    )
                
                with gradcam_tab2:
                    # Display raw heatmap
                    fig, ax = plt.subplots()
                    ax.imshow(thresholded_heatmap, cmap='jet')
                    ax.axis('off')
                    st.pyplot(fig)
        else:
            # Display just the original image with controlled width
            center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
            with center_col2:
                st.image(
                    current_image_buf,
                    caption=f"{image_name} (Image {st.session_state.image_index + 1} of {num_images})",
                    use_container_width=True,
                )

        # Mock probability output
        probabilities = np.random.dirichlet(np.ones(9), size=1)[0]  # Simulate model output
        predicted_vss = np.argmax(probabilities) + 1

        # Diagnosis Mapping
        diagnosis_labels = ["Normal", "Pre-Plus", "Plus"]
        predicted_diagnosis = np.random.choice(diagnosis_labels, p=[0.5, 0.3, 0.2])  # Simulated output

        # Show heatmap in a container with controlled width
        st.subheader("Predicted Vascular Severity Score")
        vss_col1, vss_col2, vss_col3 = st.columns([1, 6, 1])
        with vss_col2:
            fig = generate_heatmap(probabilities)
            st.pyplot(fig)

        # Show Predicted VSS and Plus Disease Diagnosis in one row
        metric_col1, metric_col2 = st.columns([1, 1])
        with metric_col1:
            st.metric(label="Predicted VSS", value=predicted_vss)
        with metric_col2:
            st.metric(label="Plus Disease Diagnosis", value=predicted_diagnosis)


if __name__ == "__main__":
    main()