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
    # Google Drive file IDs for each model
    drive_links = {
        "🌀 GRU": "1iLV7JTUcMUPskJJbtbu5Y9hP0dYflRAn",
        "🎨 Mini-VGG": "1Ep4GjnmGXJGrwEY-uLF5m0LSBIBwkjUW",
        "🧩 MLP": "1bRGZJORSvcMTYwIUlkBCBQA50kptOnZP",
        "🛠 ResNet-like": "1B__-ApSUXav_8kprT0gjHFJBZBye-AVf",
        "📘 CNN": "14WRD1cuMby2bNvtloCfH6ACsvbQKeK1V",
        "🏛 VGG-like": "1B0Vb8BsR4mVhnGfuQqe05vHSkVK56H2g",
    }

    models = {}
    os.makedirs("models", exist_ok=True)

    for name, file_id in drive_links.items():
        model_path = f"models/{name.replace(' ', '_').replace('(', '').replace(')', '')}.keras"
        
        if not os.path.exists(model_path):
            # Download model from Google Drive
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
        
        models[name] = tf.keras.models.load_model(model_path)

    # Load Tulu → Kannada mapping
    with open("tulu_to_kannada_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    class_labels = list(mapping.keys())

    return models, mapping, class_labels

all_models, character_mapping, class_labels = load_all_models()

# ---------------- Preprocessing ----------------
def preprocess_image_array(img_array, img_size=(50, 50)):
    """Preprocess: grayscale, resize, normalize"""
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

    img_array = cv2.resize(img_array, img_size)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- Prediction Function ----------------
def predict_character(image_array, model):
    img = preprocess_image_array(image_array)
    preds = model.predict(img, verbose=0)[0]
    pred_index = np.argmax(preds)
    predicted_folder = class_labels[pred_index]
    kannada_char = character_mapping[predicted_folder]
    confidence = preds[pred_index] * 100
    return kannada_char, confidence

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
    st.image("Consonant_Vowels.jpg", caption="📖 Consonants + Vowels", use_container_width=True)

with col_center:
    selected_model_name = st.selectbox("🔍 Select Model for Prediction", list(all_models.keys()))
    selected_model = all_models[selected_model_name]

    option = st.radio("✏ Input Method:", ["📤 Upload Image", "✍ Draw Character", "🌐 Image Link"])

    # -------- Upload Image --------
    if option == "📤 Upload Image":
        uploaded_file = st.file_uploader("Upload a character image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            st.image(img, caption="Uploaded Image", use_container_width=True, channels="GRAY")
            if st.button("🚀 Predict from Uploaded Image"):
                kannada_char, confidence = predict_character(img, selected_model)
                st.markdown(f"<div class='prediction-box'>Model: <b>{selected_model_name}</b><br>"
                            f"Predicted Kannada Character: <b>{kannada_char}</b><br>"
                            f"Confidence: <b>{confidence:.2f}%</b></div>", unsafe_allow_html=True)

    # -------- Draw Character --------
    elif option == "✍ Draw Character":
        st.write("🎨 Draw the character below (white pen on blackboard):")
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
                kannada_char, confidence = predict_character(img, selected_model)
                st.markdown(f"<div class='prediction-box'>Model: <b>{selected_model_name}</b><br>"
                            f"Predicted Kannada Character: <b>{kannada_char}</b><br>"
                            f"Confidence: <b>{confidence:.2f}%</b></div>", unsafe_allow_html=True)

    # -------- Image URL --------
    elif option == "🌐 Image Link":
        img_url = st.text_input("Enter the image URL:")
        if img_url:
            try:
                response = requests.get(img_url)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = np.array(img)

                # Convert to grayscale
                if img.shape[2] == 4:  # RGBA
                    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                gray = gray.astype("uint8")
                inverted = cv2.bitwise_not(gray)

                st.image(inverted, caption="Inverted Image from URL", use_container_width=True, channels="GRAY")

                if st.button("🚀 Predict from URL Image"):
                    kannada_char, confidence = predict_character(inverted, selected_model)
                    st.markdown(f"<div class='prediction-box'>Model: <b>{selected_model_name}</b><br>"
                                f"Predicted Kannada Character: <b>{kannada_char}</b><br>"
                                f"Confidence: <b>{confidence:.2f}%</b></div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠ Could not process image from URL. Error: {e}")

with col_right:
    st.image("Conjunct_Characters.jpeg", caption="📖 Conjunct Characters", use_container_width=True)
