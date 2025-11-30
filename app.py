
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# =========================
# Load Model
# =========================
model = load_model("wayang_mobilenetv2.h5")

# Class labels (urutan harus sama seperti training!)
class_labels = [
    "arjuna",
    "bagong",
    "bathara surya",
    "bathara wisnu",
    "gareng",
    "nakula",
    "petruk",
    "sadewa",
    "semar",
    "werkudara",
    "yudistira"
]

st.title("🎭 Klasifikasi Tokoh Wayang Menggunakan MobileNetV2")
st.write("Upload gambar wayang untuk diprediksi tokohnya.")

uploaded_file = st.file_uploader("📤 Upload gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    st.image(uploaded_file, caption="Gambar diupload", use_column_width=True)

    # Preprocess image
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    pred_idx = np.argmax(pred)
    pred_label = class_labels[pred_idx]
    confidence = np.max(pred) * 100

    st.subheader("✨ Hasil Prediksi:")
    st.write(f"**Tokoh Wayang:** {pred_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")
