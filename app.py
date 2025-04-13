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
            phone_numbers.append(text)
            boxes.append((
                text,
                data['left'][i],
                data['top'][i],
                data['width'][i],
                data['height'][i]
            ))

    if not phone_numbers:
        st.warning("No phone numbers detected.")
    else:
        st.image(image, caption="Original Image", use_column_width=True)

        selected_number = st.selectbox("Select the phone number to replace", phone_numbers)
        new_number = st.text_input("Enter the new number to insert")

        # New UI Controls
        font_size_input = st.slider("Font Size", min_value=10, max_value=100, value=30)
        text_color = st.color_picker("Pick Text Color", "#000000")  # Default to black

        if st.button("Replace Number"):
            for number, x, y, w, h in boxes:
                if number == selected_number:
                    # Inpaint area
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                    image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

                    # Draw new number
                    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(image_pil)

                    try:
                        font = ImageFont.truetype("arial.ttf", size=font_size_input)
                    except:
                        font = ImageFont.load_default()

                    # Convert hex to RGB tuple
                    color_rgb = tuple(int(text_color[i:i+2], 16) for i in (1, 3, 5))

                    draw.text((x, y), new_number, fill=color_rgb, font=font)
                    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                    break

            st.success("Phone number replaced successfully!")
            st.image(image, caption="Updated Image", use_column_width=True)

            _, buffer = cv2.imencode(".png", image)
            st.download_button(
                "📥 Download Updated Image",
                buffer.tobytes(),
                "updated_image.png",
                "image/png"
            )
