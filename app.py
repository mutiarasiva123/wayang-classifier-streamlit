import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# =================================================================
# PAGE SETTINGS
# =================================================================
st.set_page_config(
    page_title="Klasifikasi Tokoh Wayang",
    page_icon="🎭",
    layout="wide"
)

# =================================================================
# LOAD IMAGE AS BASE64 (LOCAL FILE)
# =================================================================
def load_base64(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

batik_base64 = load_base64("batik.png.png")

# =================================================================
# CUSTOM DARK THEME + BATIK BACKGROUND
# =================================================================
custom_css = f"""
<style>

html, body, [class*="css"] {{
    background: linear-gradient(135deg, #0f0f17 0%, #1d1b27 50%, #2b2a3a 100%) !important;
    color: #f5f5f5 !important;
}}

.main {{
    background-image: url("data:image/png;base64,{batik_base64}");
    background-size: 600px;
    background-repeat: repeat;
    padding: 30px;
    border-radius: 12px;
}}

h1, h2, h3 {{
    text-align: center;
    font-weight: 800;
    color: #fefefe !important;
    text-shadow: 0px 0px 8px #000;
}}

.upload-section {{
    background: rgba(255,255,255,0.06);
    padding: 25px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(6px);
}}

.result-box {{
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 12px;
    margin-top: 20px;
    border: 1px solid rgba(255,255,255,0.15);
}}

.button-analyze {{
    width: 100%;
    background: #6c4dd9;
    color: white !important;
    border-radius: 10px;
    font-size: 18px;
    padding: 12px;
}}

.button-analyze:hover {{
    background: #8a6aff;
    color: white !important;
}}

img {{
    border-radius: 12px;
}}

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# =================================================================
# LOAD MODEL
# =================================================================
MODEL_PATH = "wayang_mobilenetv2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

labels = ["arjuna", "bima", "gatotkaca", "nakula", "sadewa", "semar", "werkudara", "yudistira"]

# Deskripsi Wayang
deskripsi = {
    "semar": "Semar adalah punakawan tertua, bijaksana, dan pelindung Pandawa dalam pewayangan Jawa.",
    "arjuna": "Arjuna adalah ksatria tampan, ahli panah, dan tokoh penting Pandawa.",
    "bima": "Bima (Werkudara) adalah kesatria kuat, jujur, dan berwatak keras.",
    "gatotkaca": "Gatotkaca adalah ksatria sakti mandraguna, mampu terbang, anak Bima.",
    "nakula": "Nakula adalah salah satu Pandawa kembar, dikenal tampan dan ahli pedang.",
    "sadewa": "Sadewa adalah saudara kembar Nakula, terkenal bijaksana.",
    "werkudara": "Werkudara adalah nama lain Bima, simbol kekuatan dan keberanian.",
    "yudistira": "Yudistira adalah pemimpin Pandawa yang adil dan jujur."
}

# =================================================================
# UI – HEADER
# =================================================================
st.markdown("<h1>🎭 Klasifikasi Tokoh Wayang</h1>", unsafe_allow_html=True)
st.markdown("<h3>Tema Gelap • Background Batik • Gradasi Elegan</h3>", unsafe_allow_html=True)
st.write("")

# =================================================================
# UPLOAD GAMBAR
# =================================================================
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
uploaded = st.file_uploader("📤 Upload gambar tokoh wayang", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2])

# =================================================================
# PROSES GAMBAR
# =================================================================
if uploaded:
    with col1:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar yang diupload", use_container_width=True)

    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    if st.button("🔮 Analisis Gambar", use_container_width=True):
        prediction = model.predict(img_array)
        idx = np.argmax(prediction)
        tokoh = labels[idx]
        confidence = prediction[0][idx] * 100

        with col2:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.subheader(f"✨ Hasil Prediksi: **{tokoh.upper()}**")
            st.markdown(f"**Akurasi:** {confidence:.2f}%")
            st.write("---")
            st.write(f"📜 **Deskripsi Tokoh:**")
            st.write(deskripsi.get(tokoh, "Deskripsi belum tersedia."))
            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Silakan upload gambar terlebih dahulu.")

