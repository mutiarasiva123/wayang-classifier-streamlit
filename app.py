import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================================================================
# CONFIG
# =====================================================================
st.set_page_config(
    page_title="Klasifikasi Tokoh Wayang",
    page_icon="🎭",
    layout="centered"
)

# =====================================================================
# SUPER CUSTOM CSS (BATIK + MODERN UI)
# =====================================================================
st.markdown("""
<style>

    /* ====== GLOBAL BACKGROUND ====== */
    body {
        background-color: #f5f3f0;
    }

    /* ====== HEADER BATIK ====== */
    .hero {
        width: 100%;
        padding: 70px 20px;
        background-image: url('https://i.ibb.co/syjmXbL/batik-header-purple.jpg');
        background-size: cover;
        background-position: center;
        border-radius: 0px 0px 25px 25px;
        position: relative;
        text-align: center;
        color: white;
    }

    .hero::before {
        content: "";
        position: absolute;
        left:0; top:0; right:0; bottom:0;
        background: rgba(0,0,0,0.55);
        border-radius: 0px 0px 25px 25px;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        z-index: 10;
        position: relative;
    }

    .hero-sub {
        font-size: 1.3rem;
        font-weight: 300;
        margin-top: -10px;
        z-index: 10;
        position: relative;
    }

    /* ====== UPLOAD BOX CUSTOM ====== */
    .uploadbox {
        background: white;
        padding: 25px;
        border-radius: 18px;
        margin-top: 25px;
        box-shadow: 0 3px 20px rgba(0,0,0,0.12);
    }

    /* ====== IMAGE FRAME (BATIK BORDER) ====== */
    .img-frame {
        padding: 12px;
        border-radius: 15px;
        background-image: url('https://i.ibb.co/dPBJz2m/batik-border.png');
        background-size: 180px;
    }

    /* ====== RESULT CARD ====== */
    .result-card {
        margin-top: 30px;
        background: white;
        border-radius: 18px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }

    .result-title {
        font-size: 1.8rem;
        color: #3B1E54;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .confidence {
        font-weight: 600;
        color: #6A4FA3;
    }

    /* ====== BUTTON ====== */
    .stButton>button {
        background: linear-gradient(135deg, #4d2c7a, #8a4de8);
        color: white;
        border: none;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 12px 22px;
        border-radius: 12px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #341d54, #6b37b8);
        color: white;
    }

</style>
""", unsafe_allow_html=True)

# =====================================================================
# DESKRIPSI TOKOH
# =====================================================================
deskripsi_wayang = {
    "arjuna": "Arjuna adalah ksatria Pandawa yang paling tampan dan ahli memanah. Ia identik dengan ketenangan dan kebijaksanaan.",
    "bagong": "Bagong adalah punakawan lucu dan spontan yang melambangkan suara rakyat dan kritik sosial.",
    "bathara surya": "Dewa matahari pemberi terang dan kehidupan. Banyak tokoh memperoleh kesaktian darinya.",
    "bathara wisnu": "Dewa pemelihara alam semesta, penuh welas asih dan penjaga keseimbangan dunia.",
    "gareng": "Gareng adalah punakawan berhati baik, bijaksana, dan simbol kesederhanaan.",
    "nakula": "Nakula adalah salah satu kembar Pandawa, penuh disiplin dan sangat ahli dalam berkuda.",
    "petruk": "Petruk adalah punakawan tinggi jenaka yang sering mengkritik perilaku tokoh-tokoh lewat humor.",
    "sadewa": "Sadewa adalah kembaran Nakula, memiliki kecerdasan dan spiritualitas yang tinggi.",
    "semar": "Semar adalah punakawan tertua dan paling sakti. Ia sebenarnya dewa yang turun ke bumi untuk membimbing ksatria.",
    "werkudara": "Werkudara (Bima) adalah Pandawa paling kuat, jujur, dan pemberani. Ia memiliki kuku sakti Pancanaka.",
    "yudistira": "Yudistira adalah pemimpin Pandawa, simbol kejujuran dan kebijaksanaan sejati."
}

# =====================================================================
# LOAD MODEL
# =====================================================================
model = tf.keras.models.load_model("wayang_mobilenetv2.h5")
class_names = list(deskripsi_wayang.keys())


# =====================================================================
# HEADER (HERO SECTION)
# =====================================================================
st.markdown("""
<div class="hero">
    <div class="hero-title">🎭 Klasifikasi Tokoh Wayang</div>
    <div class="hero-sub">Unggah gambar wayang & dapatkan identitas serta ceritanya</div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# UPLOAD BOX
# =====================================================================
st.markdown("<div class='uploadbox'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload Gambar Tokoh Wayang", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# PROCESS IMAGE
# =====================================================================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
    st.image(image, width=320)
    st.markdown("</div>", unsafe_allow_html=True)

    img_resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    if st.button("🔍 Analisis Gambar"):
        with st.spinner("Menganalisis gambar..."):
            pred = model.predict(img_array)
            idx = np.argmax(pred)
            label = class_names[idx]
            confidence = pred[0][idx] * 100

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        st.markdown(f"<div class='result-title'>✨ Tokoh: {label.upper()}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='confidence'>📊 Keyakinan Model: {confidence:.2f}%</div>", unsafe_allow_html=True)
        st.write("---")
        st.write("### 📝 Deskripsi Singkat:")
        st.write(deskripsi_wayang[label])

        st.markdown("</div>", unsafe_allow_html=True)
