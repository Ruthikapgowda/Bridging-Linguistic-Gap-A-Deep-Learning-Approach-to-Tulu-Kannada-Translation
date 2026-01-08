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
        model_path = f"models/{name.replace(' ', '_')}.keras"
        if not os.path.exists(model_path):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
        models[name] = tf.keras.models.load_model(model_path)

    with open("tulu_to_kannada_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)

    class_labels = list(mapping.keys())
    return models, mapping, class_labels

all_models, character_mapping, class_labels = load_all_models()

# ---------------- Preprocessing ----------------
def preprocess_image_array(img_array, img_size=(50, 50)):
    img_array = cv2.resize(img_array, img_size)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- Validation (IMPORTANT) ----------------
def is_valid_tulu_image(img, model):
    img = cv2.resize(img, (50, 50))

    # Binarization
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ink ratio
    ink_ratio = np.sum(binary == 0) / binary.size
    if ink_ratio < 0.02 or ink_ratio > 0.45:
        return False, None

    # Edge density
    edges = cv2.Canny(binary, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    if edge_ratio < 0.01:
        return False, None

    # Model confidence + entropy
    img_input = preprocess_image_array(img)
    preds = model.predict(img_input, verbose=0)[0]

    confidence = np.max(preds)
    entropy = -np.sum(preds * np.log(preds + 1e-9))

    if confidence < 0.80 or entropy > 2.0:
        return False, None

    return True, confidence * 100

# ---------------- Prediction ----------------
def predict_character(image_array, model):
    img = preprocess_image_array(image_array)
    preds = model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    folder = class_labels[idx]
    return character_mapping[folder]

# ---------------- UI ----------------
st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align:center'>🖋 Tulu → Kannada Character Recognition</h1>", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1.5, 2, 1.5])

with col_left:
    st.image("Consonants_Vowels.jpg", use_container_width=True)

with col_center:
    model_name = st.selectbox("Select Model", list(all_models.keys()))
    model = all_models[model_name]

    option = st.radio("Input Method", ["📤 Upload Image", "✍ Draw Character", "🌐 Image Link"])

    # -------- Upload Image --------
    if option == "📤 Upload Image":
        uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            st.image(img, channels="GRAY")

            if st.button("Predict"):
                valid, conf = is_valid_tulu_image(img, model)
                if not valid:
                    st.error("❌ Invalid image. Please upload a handwritten Tulu character.")
                else:
                    char = predict_character(img, model)
                    st.success(f"✅ Predicted Kannada Character: {char}")
                    st.info(f"Confidence: {conf:.2f}%")

    # -------- Draw --------
    elif option == "✍ Draw Character":
        canvas = st_canvas(
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            width=256,
            height=256,
            drawing_mode="freedraw",
            key="canvas"
        )

        if canvas.image_data is not None:
            img = cv2.cvtColor(canvas.image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)
            st.image(img, channels="GRAY")

            if st.button("Predict Drawing"):
                valid, conf = is_valid_tulu_image(img, model)
                if not valid:
                    st.error("❌ Drawing does not resemble a Tulu character.")
                else:
                    char = predict_character(img, model)
                    st.success(f"✅ Predicted Kannada Character: {char}")
                    st.info(f"Confidence: {conf:.2f}%")

    # -------- URL --------
    elif option == "🌐 Image Link":
        url = st.text_input("Enter image URL")
        if st.button("Predict from URL"):
            try:
                response = requests.get(url)
                img = np.array(Image.open(BytesIO(response.content)).convert("L"))
                st.image(img, channels="GRAY")

                valid, conf = is_valid_tulu_image(img, model)
                if not valid:
                    st.error("❌ URL image is not a Tulu character.")
                else:
                    char = predict_character(img, model)
                    st.success(f"✅ Predicted Kannada Character: {char}")
                    st.info(f"Confidence: {conf:.2f}%")

            except:
                st.error("⚠ Unable to load image")

with col_right:
    st.image("Conjunct_Characters.jpeg", use_container_width=True)
