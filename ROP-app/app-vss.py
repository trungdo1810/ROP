import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_heatmap(probabilities):
    st.markdown(
        """
        <style>
        .full-width-container {
            width: 100vw;
            margin-left: -5vw;
        }
        .heatmap-container {
            width: 100%;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

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

    st.markdown(
        "<div class='full-width-container'><div class='heatmap-container'>",
        unsafe_allow_html=True,
    )
    st.pyplot(fig)
    st.markdown("</div></div>", unsafe_allow_html=True)


# Streamlit UI
def main():
    st.set_page_config(layout="wide")
    st.title("Vascular Severity Assessment")

    # Initialize session state
    if "image_index" not in st.session_state:
        st.session_state.image_index = 0

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Fundus Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True
    )

    if uploaded_files:
        num_images = len(uploaded_files)

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            if st.button("⬅️ Prev") and st.session_state.image_index > 0:
                st.session_state.image_index -= 1
        with col3:
            if st.button("Next ➡️") and st.session_state.image_index < num_images - 1:
                st.session_state.image_index += 1

        # Get current image and filename
        current_image = uploaded_files[st.session_state.image_index]
        image_name = current_image.name

        # Display current image
        st.markdown(
            """
            <style>
                .image-container {
                    display: flex;
                    justify-content: center;
                }
                .image-container img {
                    max-width: 100px;
                    height: auto;
                }
            </style>
        """,
            unsafe_allow_html=True,
        )

        st.image(
            current_image,
            caption=f"{image_name} (Image {st.session_state.image_index + 1} of {num_images})",
            use_container_width=True,
        )

        # Mock probability output
        probabilities = np.random.dirichlet(np.ones(9), size=1)[
            0
        ]  # Simulate model output
        predicted_vss = np.argmax(probabilities) + 1

        # Diagnosis Mapping
        diagnosis_labels = ["Normal", "Pre-Plus", "Plus"]
        predicted_diagnosis = np.random.choice(
            diagnosis_labels, p=[0.5, 0.3, 0.2]
        )  # Simulated output

        # Show heatmap
        st.subheader("Predicted Vascular Severity Score")
        generate_heatmap(probabilities)

        # Show Predicted VSS and Plus Disease Diagnosis in one row
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric(label="Predicted VSS", value=predicted_vss)
        with col2:
            st.metric(label="Plus Disease Diagnosis", value=predicted_diagnosis)


if __name__ == "__main__":
    main()
