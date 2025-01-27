import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import uuid
import streamlit as st
import requests
import cv2

st.set_page_config(
    page_title='Computer-aid ROP Program',
    page_icon='./static/Taipei_Tech_Logo.jpg',
    layout='wide'
)

def read_file_as_image(data) -> np.ndarray:
    """Read uploaded file data as a NumPy array."""
    image = np.array(Image.open(BytesIO(data)))
    return image

def draw_bounding_boxes(image, bounding_boxes):
    """Draw bounding boxes on an image using OpenCV."""
    for box in bounding_boxes:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        label = box['label']
        confidence = box['confidence']
        color = (0, 255, 0)  # Green for bounding boxes
        thickness = 2

        # Draw the rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Add label and confidence text
        text = f"{label} {confidence:.2f}"
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(image, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)  # Text background
        cv2.putText(image, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
    return image


# Initialize the session state for the backend URL
if "img_flask_api_url" not in st.session_state:
    st.session_state.img_flask_api_url = None

# @st.dialog("Setup Back end")
def vote():
    st.markdown(
        """
        Run the backend [here](https://colab.research.google.com/drive/1WBNPhZbU7vPUhaLyD5lhpaxxVEl6NtJj?usp=sharing) and paste the Ngrok link below.
        """
    )
    link = st.text_input("Backend URL", "")
    if st.button("Save"):
        st.session_state.img_flask_api_url = "{}/OpticDisc-segmentation".format(link)  # Update ngrok URL
        st.rerun()  # Re-run the app to close the dialog

# Display the dialog only if the URL is not set
if st.session_state.img_flask_api_url is None:
    vote()

# Once the URL is set, display it or proceed with other functionality
if st.session_state.img_flask_api_url:
    st.write(f"Backend is set to: {st.session_state.img_flask_api_url}")



def main():


    st.title("ROP-diagnosis Program")
    # session_id = str(uuid.uuid4())



    # Backend API URL
    # if "img_flask_api_url" not in st.session_state:
    #     st.session_state.img_flask_api_url = 'https://4c45-104-199-247-71.ngrok-free.app/OpticDisc-segmentation'
    
    uploaded_img = st.file_uploader('__Input your image__', type=['jpg', 'jpeg', 'png'])
    example_button = st.button('Run diagnosis')
    

    # print(type(image))
    # print(image.shape)

    # cv2.imshow("image", image)
    # print("AAAAA")

    if uploaded_img and example_button:

        image_bytes = uploaded_img.read()
        uploaded_img.seek(0)
        image = read_file_as_image(image_bytes)
        # Send the POST request to the Flask API
        files = {'image': uploaded_img}
        # print(f"Uploaded image type: {type(uploaded_img)}")

        # Use 'POST' method to send the image file to the backend ngrok
        response = requests.post(st.session_state.img_flask_api_url, files=files)

        # Check if the request is successful
        if response.status_code == 200:
            api_response = response.json()
            # segmentation_masks = api_response.get("segmentation_masks", [])

            print(type(api_response))
            print(api_response['bounding_boxes'])
            print('#HEHEHEHE'*10)

            image = draw_bounding_boxes(image, api_response["bounding_boxes"])

            # Assume api_response['b64_json'] is a list and we need the first item
            base64_mask_string = api_response['segmentation_masks'][0] if isinstance(api_response['segmentation_masks'], list) else api_response['segmentation_masks']
            
            # Decode the base64 image from the API response
            base64_decoded_mask = base64.b64decode(base64_mask_string)
            mask = Image.open(BytesIO(base64_decoded_mask))
            mask_np = np.array(mask)  

            # Display the image in Streamlit
            st.image(image, caption="OD Segmentation", width=1000)
            
            # # Add the assistant's response to the chat history
            # st.session_state.chat_history.append({"role": "assistant", "content": "Image generated and displayed above."})
        else:
            st.error(f"Error: {response.status_code}, {response.text}")

if __name__ == "__main__":
    main()
