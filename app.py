import streamlit as st
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re

st.set_page_config(page_title="Phone Number Replacer", layout="centered")
st.title("📞 AI Phone Number Replacer in Image")

uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    pattern = re.compile(r'(\+?\(?\d{1,4}\)?[\s.-]?\d{2,5}[\s.-]?\d{4,6})')
    phone_numbers = []
    boxes = []

    for i, text in enumerate(data['text']):
        if pattern.fullmatch(text.strip()):
            phone_numbers.append(text.strip())
            boxes.append((
                text.strip(),
                data['left'][i],
                data['top'][i],
                data['width'][i],
                data['height'][i]
            ))

    if phone_numbers:
        st.image(image, caption="Original Image", use_column_width=True)

        selected_number = st.selectbox("Select the phone number to replace", phone_numbers)
        new_number = st.text_input("Enter the new number to insert", value=selected_number)
        font_size_input = st.slider("Font Size", min_value=10, max_value=100, value=30)
        text_color = st.color_picker("Pick Text Color", "#000000")

        # Create a preview copy of the image
        preview_image = image.copy()

        for number, x, y, w, h in boxes:
            if number == selected_number:
                mask = np.zeros(preview_image.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                preview_image = cv2.inpaint(preview_image, mask, 3, cv2.INPAINT_TELEA)

                image_pil = Image.fromarray(cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image_pil)

                try:
                    font = ImageFont.truetype("arial.ttf", size=font_size_input)
                except:
                    font = ImageFont.load_default()

                color_rgb = tuple(int(text_color[i:i+2], 16) for i in (1, 3, 5))
                draw.text((x, y), new_number, fill=color_rgb, font=font)

                preview_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                break

        st.image(preview_image, caption="🔁 Live Preview", use_column_width=True)

        if st.button("✅ Apply and Download"):
            _, buffer = cv2.imencode(".png", preview_image)
            st.download_button("📥 Download Updated Image", buffer.tobytes(), "updated_image.png", "image/png")
    else:
        st.warning("No phone numbers were detected in the image.")
else:
    st.info("Please upload an image to begin.")
