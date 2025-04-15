import streamlit as st
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re

st.set_page_config(page_title="Phone Number Replacer", layout="centered")
st.title("📞 AI Phone Number Replacer in Image")

uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

def get_average_color(image, x, y, w, h):
    roi = image[y:y+h, x:x+w]
    avg_color = cv2.mean(roi)[:3]
    return tuple(int(c) for c in avg_color[::-1])

# Accepts a wide range of formats like:
# 0123 456 789 | +91-1234567890 | (0123) 456-789 | etc.
phone_pattern = re.compile(r'(\+?\(?\d{1,4}\)?[\s\-\.]?\d{2,5}[\s\-\.]?\d{2,5}[\s\-\.]?\d{0,5})')

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Convert image to RGB for PIL and better OCR compatibility
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    # Use Tesseract to extract full text with bounding boxes
    custom_config = r'--oem 3 --psm 11'
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=custom_config)

    phone_numbers = []
    boxes = []

    for i, text in enumerate(data['text']):
        if phone_pattern.search(text.strip()):
            number = phone_pattern.search(text.strip()).group()
            phone_numbers.append(number)
            boxes.append((
                number,
                data['left'][i],
                data['top'][i],
                data['width'][i],
                data['height'][i]
            ))

    if phone_numbers:
        st.image(image, caption="Original Image", use_column_width=True)

        selected_number = st.selectbox("Select the phone number to replace", phone_numbers)
        new_number = st.text_input("Enter the new number to insert", value=selected_number)

        preview_image = image.copy()

        for number, x, y, w, h in boxes:
            if number == selected_number:
                mask = np.zeros(preview_image.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                preview_image = cv2.inpaint(preview_image, mask, 3, cv2.INPAINT_TELEA)

                avg_color = get_average_color(image, x, y, w, h)
                font_size = int(h * 1.5)

                image_pil = Image.fromarray(cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image_pil)

                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()

                draw.text((x, y), new_number, fill=avg_color, font=font)

                preview_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                break

        st.image(preview_image, caption="🔁 Auto-Matched Preview", use_column_width=True)

        if st.button("✅ Apply and Download"):
            _, buffer = cv2.imencode(".png", preview_image)
            st.download_button("📥 Download Updated Image", buffer.tobytes(), "updated_image.png", "image/png")
    else:
        st.warning("No phone numbers were detected. Try adjusting the font/background contrast or try another image.")
else:
    st.info("Please upload an image to begin.")
