import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Klasifikasi Tokoh Wayang",
    page_icon="🎭",
    layout="centered",
)

# ============================
# CUSTOM CSS (BATIK THEME)
# ============================
st.markdown("""
<style>

    /* Background motif batik lembut */
    body {
        background-image: url('https://i.ibb.co/7VfY5dB/batik-light.png');
        background-size: 450px;
        background-repeat: repeat;
        opacity: 0.95;
    }

    /* Title styling */
    .title-container {
        text-align: center;
        margin-top: -40px;
        padding-top: 0px;
    }
    .title {
        font-size: 2.8rem;
        font-weight: 900;
        color: #3B1E54;
        text-shadow: 1px 1px 2px #ddd;
    }
    .subtitle {
        font-size: 1.15rem;
        color: #5A5A5A;
        margin-top: -10px;
        font-weight: 400;
    }

    /* Card styling */
    .result-card {
        background: #ffffffcc;
        padding: 22px;
        border-radius: 15px;
        border: 1px solid #d3c7e8;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
        margin-top: 20px;
    }

    /* Prediction title */
    .result-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #3B1E54;
    }

    /* Image centering */
    .uploaded-img {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }

    /* Button */
    .stButton>button {
        background-color: #3B1E54;
        color: white;
        padding: 10px 28px;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2a1440;
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# ============================
# DATABASE DESKRIPSI TOKOH
# ============================
deskripsi_wayang = {
    "arjuna": "Arjuna adalah ksatria Pandawa yang paling tampan dan ahli memanah. Ia identik dengan ketenangan, kebijaksanaan, serta perjalanan spiritual yang dalam.",
    "bagong": "Bagong adalah punakawan paling lucu dan unik, simbol kritik sosial. Ia diciptakan dari bayangan Semar sebagai perwujudan kejujuran dan spontanitas.",
    "bathara surya": "Dewa matahari yang memberi kehidupan, terang, dan energi. Banyak ksatria memperoleh kesaktian dari beliau.",
    "bathara wisnu": "Dewa pemelihara alam semesta. Digambarkan bijaksana, penuh welas asih, dan penjaga keseimbangan dunia.",
    "gareng": "Gareng adalah punakawan berhati lembut, bijaksana, dan penuh makna filosofis. Kekurangannya menggambarkan kerendahan hati.",
    "nakula": "Salah satu kembar Pandawa, penuh disiplin dan kesetiaan. Ia terkenal mahir berkuda dan berkepribadian lembut.",
    "petruk": "Petruk adalah punakawan tinggi jenaka, simbol rakyat kecil yang satir namun jujur. Sering mengkritik lewat humor halus.",
    "sadewa": "Kembaran Nakula, memiliki kecerdasan luar biasa dan kemampuan spiritual tinggi. Ia sosok yang sangat tenang dan lembut.",
    "semar": "Semar adalah punakawan tertua dan paling sakti. Ia sesungguhnya dewa yang turun ke bumi untuk membimbing ksatria menuju kebenaran.",
    "werkudara": "Werkudara (Bima) adalah Pandawa paling kuat. Sangat jujur, tegas, dan pemberani. Ia memiliki kuku sakti Pancanaka.",
    "yudistira": "Pemimpin Pandawa, simbol keadilan & kejujuran. Ia terkenal sabar, bijaksana, dan pantang berbohong."
}

# ============================
# LOAD MODEL
# ============================
model = tf.keras.models.load_model("wayang_mobilenetv2.h5")
class_names = list(deskripsi_wayang.keys())

# ============================
# HEADER
# ============================
st.markdown("""
<div class='title-container'>
    <div class='title'>🎭 Klasifikasi Tokoh Wayang</div>
    <div class='subtitle'>Unggah gambar tokoh wayang & dapatkan prediksi lengkap dengan cerita singkat</div>
</div>
""", unsafe_allow_html=True)

# ============================
# UPLOAD SECTION
# ============================
uploaded_file = st.file_uploader("Upload gambar tokoh wayang", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<div class='uploaded-img'>", unsafe_allow_html=True)
    st.image(image, width=300)
    st.markdown("</div>", unsafe_allow_html=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Analisis Gambar"):
        with st.spinner("Menganalisis gambar..."):
            predictions = model.predict(img_array)
            idx = np.argmax(predictions)
            label = class_names[idx]
            confidence = predictions[0][idx] * 100

        # ============================
        # RESULT CARD
        # ============================
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-title'>✨ Tokoh: {label.upper()}</div>", unsafe_allow_html=True)
        st.write(f"📊 Tingkat keyakinan model: **{confidence:.2f}%**")
        st.write("---")
        st.write("### 📝 Deskripsi Tokoh")
        st.write(deskripsi_wayang[label])
        st.markdown("</div>", unsafe_allow_html=True)

