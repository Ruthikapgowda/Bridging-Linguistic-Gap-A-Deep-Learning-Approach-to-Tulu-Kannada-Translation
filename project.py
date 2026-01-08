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
        "GRU": "1B0Vb8BsR4mVhnGfuQqe05vHSkVK56H2g",
        "Mini-VGG": "14WRD1cuMby2bNvtloCfH6ACsvbQKeK1V",
        "MLP": "1B__-ApSUXav_8kprT0gjHFJBZBye-AVf",
        "ResNet-like": "1bRGZJORSvcMTYwIUlkBCBQA50kptOnZP",
        "CNN": "1Ep4GjnmGXJGrwEY-uLF5m0LSBIBwkjUW",
        "VGG-like": "1iLV7JTUcMUPskJJbtbu5Y9hP0dYflRAn",
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

# ---------------- Preprocessing ----------------
def preprocess_image_array(img_array, img_size=(50, 50)):
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    img_array = cv2.resize(img_array, img_size)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- Validation ----------------
def is_valid_tulu_character(img, model):
    """
    Strong validation using image structure + model confidence
    """

    # ---- Resize & binarize ----
    img_resized = cv2.resize(img, (50, 50))
    _, binary = cv2.threshold(
        img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ---- Foreground (ink) ratio ----
    foreground_ratio = np.sum(binary == 0) / binary.size
    if foreground_ratio < 0.02 or foreground_ratio > 0.45:
        return False, None

    # ---- Edge density (characters have strokes) ----
    edges = cv2.Canny(binary, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    if edge_ratio < 0.01:
        return False, None

    # ---- Model confidence & entropy ----
    img_input = preprocess_image_array(img_resized)
    preds = model.predict(img_input, verbose=0)[0]

    confidence = np.max(preds)
    entropy = -np.sum(preds * np.log(preds + 1e-9))

    # Reject uncertain predictions
    if confidence < 0.80 or entropy > 2.0:
        return False, None

    return True, confidence * 100

# ---------------- Prediction ----------------
def predict_character(image_array, model):
    img = preprocess_image_array(image_array)
    preds = model.predict(img, verbose=0)[0]
    pred_index = np.argmax(preds)
    return (
        character_mapping[class_labels[pred_index]],
        preds[pred_index] * 100,
    )

# ---------------- Streamlit UI ----------------
st.set_page_config(layout="wide")

st.markdown("""
<style>
.title { text-align: center; font-size: 32px; color: #4CAF50; }
.subtitle { text-align: center; font-size: 18px; color: #bbb; }
.prediction-box { padding: 15px; border-radius: 12px; background: #e3f2fd; color: #0d47a1; font-size: 20px; margin-top: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🖋 Tulu → Kannada Character Recognition & Translation</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Choose a model and input method below 👇</p>", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1.5, 2, 1.5])

with col_left:
    st.image("Consonants_Vowels.jpg", caption="📖 Consonants + Vowels", use_container_width=True)

with col_center:
    selected_model_name = st.selectbox("🔍 Select Model for Prediction", list(all_models.keys()))
    selected_model = all_models[selected_model_name]

    option = st.radio("✏ Input Method:", ["📤 Upload Image", "✍ Draw Character", "🌐 Image Link"])

    # -------- Upload Image --------
    if st.button("🚀 Predict from Uploaded Image"):
        is_valid, confidence = is_valid_tulu_character(img, selected_model)

        if not is_valid:
            st.error("❌ This image does not belong to the Tulu character dataset.")
        else:
            kannada_char, _ = predict_character(img, selected_model)

            st.markdown(
                f"<div class='prediction-box'>"
                f"Model: <b>{selected_model_name}</b><br>"
                f"Predicted Kannada Character: <b>{kannada_char}</b><br>"
                f"Confidence: <b>{confidence:.2f}%</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # -------- Draw Character --------
    elif option == "✍ Draw Character":
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            width=256,
            height=256,
            drawing_mode="freedraw",
            key="canvas",
        )

        if canvas_result.image_data is not None:
            img = cv2.cvtColor(canvas_result.image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)
            st.image(img, caption="Drawn Image", use_container_width=True, channels="GRAY")

            if st.button("🚀 Predict from Drawing"):
                if not is_valid_tulu_character(img):
                    st.error("❌ Please draw a valid Tulu character.")
                else:
                    kannada_char, confidence = predict_character(img, selected_model)
                    st.markdown(
                        f"<div class='prediction-box'>"
                        f"Model: <b>{selected_model_name}</b><br>"
                        f"Predicted Kannada Character: <b>{kannada_char}</b><br>"
                        f"Confidence: <b>{confidence:.2f}%</b></div>",
                        unsafe_allow_html=True,
                    )

    # -------- Image URL --------
    if st.button("🚀 Predict from URL Image"):
        is_valid, confidence = is_valid_tulu_character(inverted, selected_model)

        if not is_valid:
            st.error("❌ This image does not belong to the Tulu character dataset.")
        else:
            kannada_char, _ = predict_character(inverted, selected_model)

            st.markdown(
                f"<div class='prediction-box'>"
                f"Model: <b>{selected_model_name}</b><br>"
                f"Predicted Kannada Character: <b>{kannada_char}</b><br>"
                f"Confidence: <b>{confidence:.2f}%</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

with col_right:
    st.image("Conjunct_Characters.jpeg", caption="📖 Conjunct Characters", use_container_width=True)



