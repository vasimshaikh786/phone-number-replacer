import streamlit as st
import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import re
import io

st.set_page_config(page_title="Phone Number Replacer", layout="centered")
st.title("📞 AI Phone Number Replacer in Image")

uploaded_file = st.file_uploader("Upload an image (JPG, PNG, BMP, TIFF, WEBP)", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"])

def enhance_image(image_pil):
    enhancer = ImageEnhance.Contrast(image_pil)
    image_pil = enhancer.enhance(2.0)
    enhancer = ImageEnhance.Sharpness(image_pil)
    image_pil = enhancer.enhance(2.0)
    enhancer = ImageEnhance.Brightness(image_pil)
    image_pil = enhancer.enhance(1.2)
    return image_pil

def preprocess_image(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def get_text_data(image):
    data_results = []
    configs = [
        '--oem 3 --psm 6',
        '--oem 3 --psm 11',
        '--oem 3 --psm 3',
    ]
    for config in configs:
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        if any(text.strip() for text in data['text']):
            data_results.append(data)
    return data_results[0] if data_results else None

def extract_numbers(data):
    pattern = re.compile(r'(\+?\d{0,4}[\s.-]?[\(]?\d{1,5}[\)]?[\s.-]?\d{2,6}[\s.-]?\d{2,6}[\s.-]?\d{0,6})')
    numbers = []
    boxes = []
    for i, text in enumerate(data['text']):
        clean = text.strip()
        if pattern.fullmatch(clean) and len(clean.replace(' ', '').replace('-', '').replace('(', '').replace(')', '').replace('.', '')) >= 7:
            numbers.append(clean)
            boxes.append({
                'text': clean,
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
                'conf': float(data['conf'][i])
            })
    return numbers, boxes

def get_font_metrics(image_cv, x, y, w, h):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    roi = gray[y:y+h, x:x+w]
    mean_val = np.mean(roi)
    font_color = (255, 255, 255) if mean_val < 128 else (0, 0, 0)
    return font_color

def format_number(original, new):
    if re.match(r'\+1\s\(\d{3}\)\s\d{3}-\d{4}', original):
        return f"+1 ({new[:3]}) {new[3:6]}-{new[6:]}"
    return new

if uploaded_file:
    try:
        image_pil = Image.open(uploaded_file).convert('RGB')
        cv_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        enhanced_pil = enhance_image(image_pil)
        enhanced_cv = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
        preprocessed = preprocess_image(enhanced_cv)

        ocr_data = get_text_data(preprocessed)
        if ocr_data:
            phone_numbers, boxes = extract_numbers(ocr_data)
        else:
            phone_numbers, boxes = [], []

        if phone_numbers:
            st.image(cv_image, caption="Original Image", use_column_width=True)
            selected_number = st.selectbox("Select phone number to replace", phone_numbers)
            new_number_raw = st.text_input("Enter new number (10 digits)", value=selected_number.replace(' ', '').replace('-', '').replace('(', '').replace(')', ''))
            new_number = format_number(selected_number, new_number_raw.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')[:10])
            
            # Add font size slider
            default_font_size = max(10, int(boxes[0]['height'] * 0.8))
            font_size = st.slider("Adjust Font Size", min_value=10, max_value=50, value=default_font_size, step=1)

            preview = cv_image.copy()
            for box in boxes:
                if box['text'] == selected_number:
                    x, y, w, h = box['left'], box['top'], box['width'], box['height']
                    
                    # Inpaint to remove original number
                    mask = np.zeros(preview.shape[:2], dtype=np.uint8)
                    cv2.rectangle(mask, (x-2, y-2), (x+w+2, y+h+2), 255, -1)
                    preview = cv2.inpaint(preview, mask, 3, cv2.INPAINT_TELEA)

                    # Calculate font metrics
                    font_color = get_font_metrics(cv_image, x, y, w, h)
                    
                    # Draw new number
                    img_pil = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
                    
                    # Calculate text size for centering
                    text_bbox = draw.textbbox((0, 0), new_number, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    text_x = x + (w - text_width) // 2
                    text_y = y + (h - text_height) // 2
                    
                    draw.text((text_x, text_y), new_number, fill=font_color, font=font)
                    preview = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    break

            st.image(preview, caption="🔁 Preview with Replaced Number", use_column_width=True)

            if st.button("✅ Apply and Download"):
                _, buffer = cv2.imencode(".png", preview)
                st.download_button(
                    "📥 Download Updated Image",
                    buffer.tobytes(),
                    "updated_image.png",
                    "image/png"
                )
        else:
            st.warning("❌ No phone numbers were detected. Try another image or clearer version.")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.info("⬆️ Please upload an image to begin.")
