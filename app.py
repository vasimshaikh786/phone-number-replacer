import streamlit as st
import pytesseract
import easyocr
import cv2
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Initialize Streamlit
st.set_page_config(page_title="📞 Smart Phone Number Replacer", layout="centered")
st.title("📞 AI Smart Phone Number Replacer")

# Helper functions

def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    return image

def detect_with_tesseract(img):
    img_cv = np.array(img)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img_gray, config=custom_config)
    return text

def detect_with_easyocr(img):
    reader = easyocr.Reader(['en'], gpu=False)
    img_cv = np.array(img)
    results = reader.readtext(img_cv)
    text = " ".join([res[1] for res in results])
    return text

def extract_phone_numbers(text):
    phone_regex = re.compile(r'''
        (\+?\d{1,3}[\s\.\-]?)?        # Country code
        (\(?\d{2,4}\)?[\s\.\-]?)?     # Area code
        (\d{2,4}[\s\.\-]?){2,3}       # Main number
    ''', re.VERBOSE)

    matches = []
    for match in phone_regex.finditer(text):
        if match.group().strip():
            matches.append(match.group().strip())
    return matches

def match_format(original, new):
    # Try to preserve special characters spacing etc.
    formatted = ""
    digits = re.sub(r'\D', '', new)
    digit_idx = 0

    for ch in original:
        if ch.isdigit():
            if digit_idx < len(digits):
                formatted += digits[digit_idx]
                digit_idx += 1
            else:
                formatted += "0"
        else:
            formatted += ch

    return formatted

def find_text_position(img, target):
    img_cv = np.array(img)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    data = pytesseract.image_to_data(img_gray, output_type=pytesseract.Output.DICT)

    for i in range(len(data['text'])):
        text = data['text'][i]
        if target.replace(" ", "") in text.replace(" ", ""):
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            return (x, y, w, h)

    return None

def replace_phone_number(img, old_number, new_number):
    img_cv = np.array(img)
    pos = find_text_position(img, old_number)

    if pos is None:
        return img_cv, False

    (x, y, w, h) = pos

    # Estimate font size and color
    font_size = int(h * 0.8)
    color = img_cv[y+h//2, x+w//2].tolist()

    # Draw white rectangle
    cv2.rectangle(img_cv, (x, y), (x+w, y+h), (255, 255, 255), -1)

    # Draw new text
    img_pil = Image.fromarray(img_cv)
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    draw.text((x, y), new_number, fill=tuple(color), font=font)
    img_cv = np.array(img_pil)

    return img_cv, True

# Streamlit app

uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = load_image(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Step 1: Detect Phone Numbers")

    method = st.radio("Choose OCR method:", ["Tesseract (fast)", "EasyOCR (more accurate)"])

    if method == "Tesseract (fast)":
        text = detect_with_tesseract(image)
    else:
        text = detect_with_easyocr(image)

    numbers = extract_phone_numbers(text)

    if numbers:
        st.success(f"Detected {len(numbers)} phone number(s).")
        selected_number = st.selectbox("Select number to replace:", numbers)
        new_number_raw = st.text_input("Enter new phone number:")

        if st.button("Replace Number"):
            if new_number_raw.strip() == "":
                st.error("Please enter a valid new number.")
            else:
                # Match format
                new_number = match_format(selected_number, new_number_raw)
                output_img, success = replace_phone_number(image, selected_number, new_number)

                if success:
                    st.image(output_img, caption="Modified Image", use_column_width=True)
                    st.success(f"Phone number replaced: `{selected_number}` ➔ `{new_number}`")

                    output_pil = Image.fromarray(output_img)
                    output_pil.save("output_advanced.png")
                    with open("output_advanced.png", "rb") as file:
                        st.download_button(
                            label="Download Image",
                            data=file,
                            file_name="replaced_advanced.png",
                            mime="image/png"
                        )
                else:
                    st.error("Could not locate phone number position clearly.")
    else:
        st.error("No phone numbers were detected. Try another image or OCR method.")
