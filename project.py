import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
import requests
from io import BytesIO
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import gdown
import os

# ---------------- Load Models + Mapping ----------------
@st.cache_resource
def load_all_models():
    drive_links = {
        "🌀 GRU": "1B0Vb8BsR4mVhnGfuQqe05vHSkVK56H2g",
        "🎨 Mini-VGG": "14WRD1cuMby2bNvtloCfH6ACsvbQKeK1V",
        "🧩 MLP": "1B__-ApSUXav_8kprT0gjHFJBZBye-AVf",
        "🛠 ResNet-like": "1bRGZJORSvcMTYwIUlkBCBQA50kptOnZP",
        "📘 CNN": "1Ep4GjnmGXJGrwEY-uLF5m0LSBIBwkjUW",
        "🏛 VGG-like": "1iLV7JTUcMUPskJJbtbu5Y9hP0dYflRAn",
    }

    models = {}
    os.makedirs("models", exist_ok=True)

    for name, file_id in drive_links.items():
        model_path = f"models/{name.replace(' ', '_').replace('(', '').replace(')', '')}.keras"
        if not os.path.exists(model_path):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
        models[name] = tf.keras.models.load_model(model_path)

    with open("tulu_to_kannada_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)

    class_labels = list(mapping.keys())
    return models, mapping, class_labels


all_models, character_mapping, class_labels = load_all_models()

# ---------------- Image Normalization ----------------
def normalize_character_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (50, 50))

    # Auto invert (handle white background)
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    return img


# ---------------- Validation + Prediction ----------------
def is_valid_tulu_character(img, model, confidence_threshold=0.60):
    if img is None or img.size == 0:
        return False, 0.0, None

    img = normalize_character_image(img)

    # Binary-like check
    if len(np.unique(img)) > 80:
        return False, 0.0, None

    # Stroke density check
    white_ratio = np.sum(img > 200) / img.size
    if white_ratio < 0.01 or white_ratio > 0.65:
        return False, 0.0, None

    img_input = img.astype("float32") / 255.0
    img_input = np.expand_dims(img_input, axis=(0, -1))

    preds = model.predict(img_input, verbose=0)[0]
    confidence = np.max(preds)

    if confidence < confidence_threshold:
        return False, confidence * 100, None

    return True, confidence * 100, img


def predict_character(img, model):
    img_input = img.astype("float32") / 255.0
    img_input = np.expand_dims(img_input, axis=(0, -1))
    preds = model.predict(img_input, verbose=0)[0]
    index = np.argmax(preds)
    folder = class_labels[index]
    return character_mapping[folder]


# ---------------- Streamlit UI ----------------
st.set_page_config(layout="wide")

st.markdown("""
<style>
.title { text-align: center; font-size: 32px; color: #4CAF50; }
.subtitle { text-align: center; font-size: 18px; color: #bbb; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🖋 Tulu → Kannada Character Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Robust validation for handwritten Tulu characters</p>", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1.5, 2, 1.5])

with col_left:
    st.image("Consonants_Vowels.jpg", caption="📖 Consonants & Vowels", use_container_width=True)

with col_center:
    model_name = st.selectbox("Select Model", list(all_models.keys()))
    model = all_models[model_name]

    option = st.radio("Input Method", ["📤 Upload Image", "✍ Draw Character", "🌐 Image Link"])

    # -------- Upload --------
    if option == "📤 Upload Image":
        file = st.file_uploader("Upload character image", type=["png", "jpg", "jpeg"])
        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            st.image(img, caption="Uploaded Image", channels="GRAY")

            if st.button("🚀 Predict"):
                valid, conf, processed = is_valid_tulu_character(img, model)
                if not valid:
                    st.error("❌ This image does not match the trained Tulu character format.")
                else:
                    char = predict_character(processed, model)
                    st.success(f"Predicted Kannada Character: {char}")
                    st.info(f"Confidence: {conf:.2f}%")

    # -------- Draw --------
    elif option == "✍ Draw Character":
        canvas = st_canvas(
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            width=256,
            height=256,
            drawing_mode="freedraw"
        )

        if canvas.image_data is not None:
            img = cv2.cvtColor(canvas.image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)
            st.image(img, caption="Drawn Image", channels="GRAY")

            if st.button("🚀 Predict"):
                valid, conf, processed = is_valid_tulu_character(img, model)
                if not valid:
                    st.error("❌ Drawn image is not a valid Tulu character.")
                else:
                    char = predict_character(processed, model)
                    st.success(f"Predicted Kannada Character: {char}")
                    st.info(f"Confidence: {conf:.2f}%")

    # -------- URL --------
    else:
        url = st.text_input("Enter image URL")
        if url:
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                st.image(img, caption="URL Image", channels="GRAY")

                if st.button("🚀 Predict"):
                    valid, conf, processed = is_valid_tulu_character(img, model)
                    if not valid:
                        st.error("❌ URL image is not a valid Tulu character.")
                    else:
                        char = predict_character(processed, model)
                        st.success(f"Predicted Kannada Character: {char}")
                        st.info(f"Confidence: {conf:.2f}%")

            except:
                st.error("⚠ Unable to load image from URL.")

with col_right:
    st.image("Conjunct_Characters.jpeg", caption="📖 Conjunct Characters", use_container_width=True)
