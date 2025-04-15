import streamlit as st
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re

st.set_page_config(page_title="Phone Number Replacer", layout="centered")
st.title("📞 AI Phone Number Replacer in Image")

uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

# Function to get average RGB color from a region
def get_average_color(image, x, y, w, h):
    roi = image[y:y+h, x:x+w]
    avg_color = cv2.mean(roi)[:3]  # BGR
    return tuple(int(c) for c in avg_color[::-1])  # convert to RGB

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    pattern = re.compile(r'(\+?\(?\d{1,4}\)?[\s.-]?\d{2,5}[\s.-]?\d{4,6})')
    phone_numbers = []
    boxes = []

    for i, text in enumerate(data['text']):
        if
