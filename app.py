import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================
#   CUSTOM UI / THEME
# ============================
st.set_page_config(
    page_title="Klasifikasi Tokoh Wayang",
    page_icon="🎭",
    layout="centered"
)

# Custom CSS untuk mempercantik UI
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: bold;
        color: #4B2E83;
        margin-bottom: -10px;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    .footer {
        text-align:center;
        margin-top: 50px;
        font-size: 0.8rem;
        color: #777;
    }
    .stButton>button {
        background-color: #4B2E83;
        color: white;
        padding: 10px 20px;
        border-radius: 7px;
        border: none;
        font-size: 1rem;
    }
    .stButton>button:hover {
        background-color: #351e5c;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ============================
#   LOAD MODEL
# ============================
model = tf.keras.models.load_model("wayang_mobilenetv2.h5")

class_names = [
    "arjuna", "bagong", "bathara surya", "bathara wisnu", "gareng",
    "nakula", "petruk", "sadewa", "semar", "werkudara", "yudistira"
]

# ============================
#   HEADER UI
# ============================
st.markdown("<div class='title'>🎭 Klasifikasi Tokoh Wayang</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload gambar wayang, dan sistem akan memprediksi tokohnya</div>", unsafe_allow_html=True)

# ============================
#   UPLOAD IMAGE
# ============================
uploaded_file = st.file_uploader("Upload gambar tokoh wayang", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Tampilkan gambar
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # PREPROCESS
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # BUTTON PREDIKSI
    if st.button("🔍 Prediksi Tokoh"):
        with st.spinner("Menganalisis gambar..."):
            predictions = model.predict(img_array)
            idx = np.argmax(predictions)
            result = class_names[idx]
            confidence = predictions[0][idx] * 100

        # OUTPUT CARD
        st.success(f"🎯 **Tokoh Wayang:** {result.capitalize()}")
        st.info(f"📊 Tingkat Kepercayaan: **{confidence:.2f}%**")

# FOOTER
st.markdown("<div class='footer'>Developed with ❤️ using Streamlit & MobileNetV2</div>", unsafe_allow_html=True)
