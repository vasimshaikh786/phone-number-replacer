import streamlit as st
import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import re

st.set_page_config(page_title="Phone Number Replacer", layout="centered")
st.title("📞 AI Phone Number Replacer in Image")

uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

def enhance_image(image_pil):
    image_pil = ImageEnhance.Contrast(image_pil).enhance(2)
    image_pil = ImageEnhance.Sharpness(image_pil).enhance(2)
    return image_pil

def get_text_data(image):
    data_results = []
    for psm in [6, 11, 3]:
        custom_config = f'--oem 3 --psm {psm}'
        data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
        if any(text.strip() for text in data['text']):
            data_results.append(data)
    return data_results[0] if data_results else None

def extract_numbers(data):
    pattern = re.compile(r'(\+?\(?\d{1,4}\)?[\s.-]?\d{2,5}[\s.-]?\d{3,6}[\s.-]?\d{0,5})')
    numbers = []
    boxes = []
    for i, text in enumerate(data['text']):
        clean = text.strip()
        if pattern.fullmatch(clean):
            numbers.append(clean)
            boxes.append((clean, data['left'][i], data['top'][i], data['width'][i], data['height'][i]))
    return numbers, boxes

def get_average_color(image, x, y, w, h):
    roi = image[y:y+h, x:x+w]
    avg_color = cv2.mean(roi)[:3]
    return tuple(int(c) for c in avg_color[::-1])  # BGR → RGB

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    cv_image = cv2.imdecode(file_bytes, 1)
    image_pil = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    enhanced = enhance_image(image_pil)
    enhanced_cv = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)

    ocr_data = get_text_data(enhanced)
    if ocr_data:
        phone_numbers, boxes = extract_numbers(ocr_data)
    else:
        phone_numbers, boxes = [], []

    if phone_numbers:
        st.image(cv_image, caption="Original Image", use_column_width=True)
        selected_number = st.selectbox("Select phone number to replace", phone_numbers)
        new_number = st.text_input("Enter new number", value=selected_number)

        preview = cv_image.copy()
        for number, x, y, w, h in boxes:
            if number == selected_number:
                mask = np.zeros(preview.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                preview = cv2.inpaint(preview, mask, 3, cv2.INPAINT_TELEA)

                avg_color = get_average_color(cv_image, x, y, w, h)
                font_size = int(h * 1.2)

                img_pil = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                draw.text((x, y), new_number, fill=avg_color, font=font)

                preview = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                break

        st.image(preview, caption="🔁 Preview with Replaced Number", use_column_width=True)

        if st.button("✅ Apply and Download"):
            _, buffer = cv2.imencode(".png", preview)
            st.download_button("📥 Download Updated Image", buffer.tobytes(), "updated_image.png", "image/png")
    else:
        st.warning("❌ No phone numbers were detected. Try another image or clearer version.")
else:
    st.info("⬆️ Please upload an image to begin.")
