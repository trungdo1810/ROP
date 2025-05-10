import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import requests
import base64

st.set_page_config(layout="wide")


def generate_heatmap(probabilities):
    """
    Generate a heatmap for the predicted probabilities.
    """
    fig, ax = plt.subplots(figsize=(10, 0.5))
    sns.heatmap(
        [probabilities],
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        cbar=False,
        xticklabels=["Normal", "Pre-Plus", "Plus"],
        yticklabels=["VSS"],
    )
    return fig


def generate_similarity_heatmap(similarities):
    """
    Generate a heatmap for the similarities to the 9 reference images.
    """
    fig, ax = plt.subplots(figsize=(10, 0.5))
    sns.heatmap(
        [list(similarities.values())],
        cmap="YlOrRd",
        annot=True,
        fmt=".4f",
        cbar=False,
        xticklabels=range(1, 10),
        yticklabels=["Similarity"],
    )
    # plt.xticks(rotation=45, ha="right")
    return fig


def overlay_heatmap(image, heatmap):
    """Create an overlay of the original image with the heatmap"""
    img_array = np.array(image)

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

    # Create colored heatmap
    cmap = plt.cm.jet
    heatmap_colored = cmap(heatmap)[:, :, :3]

    # Blend images
    alpha = 0.4
    overlay = (1 - alpha) * img_array / 255.0 + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)

    return overlay


# Dialog to input ngrok URL
@st.dialog("Setup Backend")
def setup_backend():
    st.markdown(
        """
        Paste the ngrok URL from your backend server below (e.g., https://abc123.ngrok-free.app).
        """
    )
    link = st.text_input("Backend URL", "")
    if st.button("Save"):
        if link:
            st.session_state.backend_url = f"{link}/gradcam"
            st.rerun()
        else:
            st.error("Please enter a valid URL.")


def main():
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
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = None
    if "results" not in st.session_state:
        st.session_state.results = None

    # Show dialog if backend URL is not set
    if st.session_state.backend_url is None:
        setup_backend()

    # Display backend URL if set
    if st.session_state.backend_url:
        st.write(f"Connected to backend at: {st.session_state.backend_url}")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Fundus Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True
    )

    if uploaded_files:
        num_images = len(uploaded_files)

        # Navigation buttons
        nav_col1, nav_col2, nav_col3 = st.columns([1, 6, 1])
        with nav_col1:
            if st.button("⬅️ Prev") and st.session_state.image_index > 0:
                st.session_state.image_index -= 1
                st.session_state.results = None  # Clear previous results
        with nav_col3:
            if st.button("Next ➡️") and st.session_state.image_index < num_images - 1:
                st.session_state.image_index += 1
                st.session_state.results = None  # Clear previous results

        # Current image
        current_image = uploaded_files[st.session_state.image_index]
        image_name = current_image.name
        image_bytes = current_image.getvalue()
        current_image_buf = io.BytesIO(image_bytes)
        image_pil = Image.open(current_image_buf).convert("RGB")

        # Image display section
        st.subheader("Fundus Image Analysis")

        # GradCAM toggle
        st.session_state.show_gradcam = st.checkbox(
            "Show Diagnosis", value=st.session_state.show_gradcam
        )

        # Call backend if results are not already fetched
        if st.session_state.results is None and st.session_state.show_gradcam:
            # Send request to backend
            files = {"image": (image_name, image_bytes, "image/jpeg")}
            data = {"threshold": st.session_state.threshold}
            try:
                response = requests.post(
                    st.session_state.backend_url, files=files, data=data
                )
                if response.status_code == 200:
                    st.session_state.results = response.json()
                else:
                    st.error(f"Error: {response.status_code}, {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend: {e}")

        # Initialize results
        heatmap = None
        thresholded_heatmap = None
        probabilities = None
        diagnosis = None
        similarities = None

        if st.session_state.show_gradcam and st.session_state.backend_url:
            # Threshold slider
            threshold = st.slider(
                "Attention Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.threshold,
                step=0.05,
            )
            st.session_state.threshold = threshold

            # Send request to backend
            files = {"image": (image_name, image_bytes, "image/jpeg")}
            data = {"threshold": threshold}

            # Debug: Print the image name and size
            # print(
            #     f"Sending image to backend: {image_name}, size: {len(image_bytes)} bytes"
            # )

            try:
                response = requests.post(
                    st.session_state.backend_url, files=files, data=data
                )
                if response.status_code == 200:
                    result = response.json()
                    # Decode heatmaps
                    heatmap_bytes = base64.b64decode(result["heatmap"])
                    heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
                    heatmap = np.array(heatmap_img) / 255.0

                    thresholded_bytes = base64.b64decode(result["thresholded_heatmap"])
                    thresholded_img = Image.open(io.BytesIO(thresholded_bytes))
                    thresholded_heatmap = np.array(thresholded_img) / 255.0

                    probabilities = result["probabilities"]
                    diagnosis = result["diagnosis"]
                    similarities = result["similarities"]

                    # Debug: Print updated similarities
                    print(f"Updated similarities: {similarities}")
                else:
                    st.error(f"Error: {response.status_code}, {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend: {e}")

        # Display images
        if st.session_state.show_gradcam and heatmap is not None:
            img_col1, img_col2 = st.columns([1, 1])
            with img_col1:
                st.markdown("### Original Image")
                st.image(
                    image_pil,
                    caption=f"{image_name} (Image {st.session_state.image_index + 1} of {num_images})",
                    use_container_width=True,
                )
            with img_col2:
                st.markdown("### GradCAM Visualization")
                gradcam_tab1, gradcam_tab2 = st.tabs(["Heatmap Overlay", "Raw Heatmap"])
                with gradcam_tab1:
                    overlay = overlay_heatmap(image_pil, thresholded_heatmap)
                    st.image(
                        overlay, caption="GradCAM Overlay", use_container_width=True
                    )
                with gradcam_tab2:
                    fig, ax = plt.subplots()
                    ax.imshow(thresholded_heatmap, cmap="jet")
                    ax.axis("off")
                    st.pyplot(fig)
        else:
            center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
            with center_col2:
                st.image(
                    image_pil,
                    caption=f"{image_name} (Image {st.session_state.image_index + 1} of {num_images})",
                    use_container_width=True,
                )

        # Display diagnosis and similarities
        if st.session_state.results:
            results = st.session_state.results
            probabilities = results["probabilities"]
            diagnosis = results["diagnosis"]
            similarities = results["similarities"]

            st.subheader("Predicted Vascular Severity Score")
            vss_col1, vss_col2, vss_col3 = st.columns([1, 6, 1])
            with vss_col2:
                similarity_fig = generate_similarity_heatmap(similarities)
                st.pyplot(similarity_fig)

            metric_col1, metric_col2 = st.columns([1, 1])
            with metric_col1:
                # Get the position (index) of the most similar reference
                most_similar_index = (
                    list(similarities.values()).index(max(similarities.values())) + 1
                )
                st.metric(label="VSS Prediction", value=f"{most_similar_index}")
            with metric_col2:
                st.metric(label="Plus Disease Diagnosis", value=diagnosis)


if __name__ == "__main__":
    main()
