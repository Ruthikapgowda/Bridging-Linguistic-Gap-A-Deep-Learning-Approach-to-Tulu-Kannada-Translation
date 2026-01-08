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

# ---------------- Preprocessing ----------------
def preprocess_image_array(img_array, img_size=(50, 50)):
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, img_size)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

# ---------------- Prediction ----------------
def predict_character(image_array, model):
    img = preprocess_image_array(image_array)
    preds = model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    return character_mapping[class_labels[idx]], preds[idx] * 100

# ---------------- Validation (IMPORTANT) ----------------
def is_valid_character(image_array, model, threshold=0.65):
    # Reject empty / blank images
    if image_array is None or image_array.size == 0:
        return False, 0.0

    if np.mean(image_array) > 245:
        return False, 0.0

    img = preprocess_image_array(image_array)
    preds = model.predict(img, verbose=0)[0]
    confidence = np.max(preds)

    if confidence < threshold:
        return False, confidence * 100

    return True, confidence * 100

# ---------------- Streamlit UI ----------------
st.set_page_config(layout="wide")

st.markdown("""
<style>
.title { text-align: center; font-size: 32px; color: #4CAF50; }
.subtitle { text-align: center; font-size: 18px; color: #bbb; }
.prediction-box { padding: 15px; border-radius: 12px; background: #e3f2fd;
color: #0d47a1; font-size: 20px; margin-top: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🖋 Tulu → Kannada Character Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload, Draw or Paste Image URL</p>", unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1.5, 2, 1.5])

with col_left:
    st.image("Consonants_Vowels.jpg", caption="📖 Consonants + Vowels", use_container_width=True)

with col_center:
    selected_model_name = st.selectbox("🔍 Select Model", list(all_models.keys()))
    selected_model = all_models[selected_model_name]

    option = st.radio("✏ Input Method", ["📤 Upload Image", "✍ Draw Character", "🌐 Image Link"])

    # -------- Upload Image --------
    if option == "📤 Upload Image":
        uploaded_file = st.file_uploader("Upload Tulu character image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            st.image(img, caption="Uploaded Image", use_container_width=True, channels="GRAY")

            if st.button("🚀 Predict"):
                valid, conf = is_valid_character(img, selected_model)
                if not valid:
                    st.error("❌ This image does not belong to the trained Tulu character set.")
                else:
                    char, _ = predict_character(img, selected_model)
                    st.markdown(
                        f"<div class='prediction-box'>"
                        f"Predicted Kannada Character: <b>{char}</b><br>"
                        f"Confidence: <b>{conf:.2f}%</b>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

    # -------- Draw Character --------
    elif option == "✍ Draw Character":
        canvas = st_canvas(
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            width=256,
            height=256,
            drawing_mode="freedraw",
            key="canvas",
        )

        if canvas.image_data is not None:
            img = cv2.cvtColor(canvas.image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)
            st.image(img, caption="Drawn Image", use_container_width=True, channels="GRAY")

            if st.button("🚀 Predict"):
                valid, conf = is_valid_character(img, selected_model)
                if not valid:
                    st.error("❌ Drawn image is not a valid Tulu character.")
                else:
                    char, _ = predict_character(img, selected_model)
                    st.markdown(
                        f"<div class='prediction-box'>"
                        f"Predicted Kannada Character: <b>{char}</b><br>"
                        f"Confidence: <b>{conf:.2f}%</b>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

    # -------- Image URL --------
    elif option == "🌐 Image Link":
        url = st.text_input("Enter image URL")

        if url:
            try:
                response = requests.get(url, timeout=5)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                img = cv2.bitwise_not(img)

                st.image(img, caption="URL Image", use_container_width=True, channels="GRAY")

                if st.button("🚀 Predict"):
                    valid, conf = is_valid_character(img, selected_model)
                    if not valid:
                        st.error("❌ This URL image is not a valid Tulu character.")
                    else:
                        char, _ = predict_character(img, selected_model)
                        st.markdown(
                            f"<div class='prediction-box'>"
                            f"Predicted Kannada Character: <b>{char}</b><br>"
                            f"Confidence: <b>{conf:.2f}%</b>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

            except Exception:
                st.error("⚠ Unable to process image from URL.")

with col_right:
    st.image("Conjunct_Characters.jpeg", caption="📖 Conjunct Characters", use_container_width=True)
