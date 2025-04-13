import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Title of the app
st.title("Phone Number Replacer")

# Drag and Drop to upload image
uploaded_file = st.file_uploader("Drag and drop an image here or click to upload", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Select the phone number to replace
    selected_number = st.selectbox("Select the phone number to replace", options=["Number1", "Number2", "Number3"])  # Populate with actual numbers

    # Enter the new number to insert
    new_number = st.text_input("Enter the new number to insert")

    # Font size and text color options
    font_size = st.slider("Font Size", min_value=10, max_value=100, value=30)
    text_color = st.color_picker("Select Text Color", "#000000")

    if st.button("Replace Number"):
        # Logic to replace the number
        for number, x, y, w, h in boxes:  # Assuming boxes is defined elsewhere
            if number == selected_number:
                # Inpaint the number area
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

                # Draw the new number using PIL
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image_pil)

                try:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
                except:
                    font = ImageFont.load_default()

                draw.text((x, y), new_number, fill=text_color, font=font)

                # Convert back to OpenCV image
                image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                break

        st.success("Phone number replaced successfully!")
        st.image(image, caption="Updated Image", use_container_width=True)

        # Download button to download changes option
        _, buffer = cv2.imencode(".png", image)
        st.download_button("📥 Download Updated Image", buffer.tobytes(), "updated_image.png", "image/png")

# Additional UI enhancements can be added here for better look and feel
