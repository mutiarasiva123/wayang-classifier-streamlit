import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# ================================
#  FUNGSI LOAD BACKGROUND BATIK
# ================================
def load_base64_img(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

batik_base64 = load_base64_img("batik.png")

# ================================
#  CUSTOM DARK THEME + BATIK CSS
# ================================
st.markdown(f"""
<style>

    /* WRAPPER UTAMA STREAMLIT */
    .stApp {{
        background: 
            linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.88)),
            url("data:image/png;base64,{batik_base64}");
        background-size: 140px;
        background-attachment: fixed;
        background-repeat: repeat;
        color: #eee !important;
        font-family: 'Segoe UI', sans-serif;
        padding: 0;
        margin: 0;
    }}

    /* CONTAINER KONTEN */
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 3rem;
        background: rgba(20, 15, 30, 0.55);
        border-radius: 20px;
        box-shadow: 0 0 30px rgba(0,0,0,0.55);
        backdrop-filter: blur(6px);
    }}

    /* HEADER CUSTOM */
    .hero {{
        background: linear-gradient(135deg, #2a1f49cc, #0e0b1688);
        padding: 40px 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 35px;
        border: 2px solid rgba(255,255,255,0.07);
    }}

    .title {{
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: -5px;
    }}

    .subtitle {{
        opacity: 0.9;
        font-size: 1.2rem;
        margin-top: 5px;
    }}

    /* CARD STYLE */
    .card {{
        background: rgba(255,255,255,0.07);
        padding: 22px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 25px rgba(0,0,0,0.45);
        backdrop-filter: blur(8px);
        margin: 15px 0;
        color: #ddd;
    }}

    /* UPLOADER DARK MODE */
    [data-testid="stFileUploader"] {{
        background: rgba(255,255,255,0.08);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.15);
        color: #ddd;
    }}

    /* IMAGE FRAME */
    .img-frame {{
        padding: 10px;
        border: 4px solid #7d5cd1;
        border-radius: 15px;
        background: rgba(255,255,255,0.05);
        width: 100%;
        text-align: center;
    }}

    /* MODERN BUTTON */
    .stButton > button {{
        background: linear-gradient(135deg, #8a4de8, #4d2c7a);
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: 0.2s;
        box-shadow: 0 0 15px rgba(138,77,232,0.4);
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, #6d37c5, #341f59);
        transform: scale(1.03);
        box-shadow: 0 0 25px rgba(138,77,232,0.7);
    }}

    /* HASIL PREDIKSI */
    .result-title {{
        font-size: 1.8rem;
        font-weight: 800;
        color: #d8caff;
        padding-top: 10px;
        text-align: center;
        animation: fadeIn 1s ease;
    }}

    /* ANIMASI */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

</style>
""", unsafe_allow_html=True)

# ============================================
#  LOAD MODEL
# ============================================
model = tf.keras.models.load_model("wayang_mobilenetv2.h5")

class_names = [
    "arjuna", "bagong", "bathara surya", "bathara wisnu",
    "gareng", "nakula", "petruk", "sadewa", "semar",
    "werkudara", "yudistira"
]

# ============================================
#  DATABASE DESKRIPSI WAYANG
# ============================================
wayang_info = {
    "semar": "Semar adalah tokoh punakawan tertua, lambang kebijaksanaan, kesabaran, dan pengayom manusia.",
    "bagong": "Bagong adalah anak Semar yang jenaka, humoris, dan menjadi simbol suara rakyat.",
    "gareng": "Gareng adalah punakawan yang melambangkan kehati-hatian dan moral lurus.",
    "petruk": "Petruk adalah punakawan tinggi kurus, simbol keluwesan dan kecerdikan.",
    "arjuna": "Arjuna adalah ksatria tampan, ahli panah, tokoh Pandawa yang penuh kebijaksanaan.",
    "nakula": "Nakula adalah salah satu kembar Pandawa, lambang ketampanan dan kesetiaan.",
    "sadewa": "Sadewa adalah kembaran Nakula, dikenal bijaksana dan sangat setia.",
    "yudistira": "Yudistira adalah raja bijaksana, sulung Pandawa, dikenal jujur dan adil.",
    "werkudara": "Werkudara/Bima adalah Pandawa terkuat, simbol keberanian & ketegasan.",
    "bathara surya": "Bathara Surya adalah dewa matahari dalam mitologi Jawa.",
    "bathara wisnu": "Bathara Wisnu adalah dewa pemelihara alam semesta."
}

# ============================================
#  HEADER HERO SECTION
# ============================================
st.markdown("""
<div class="hero">
    <div class="title">🔮 Klasifikasi Tokoh Wayang</div>
    <div class="subtitle">Tema Gelap • Background Batik • Desain Elegan</div>
</div>
""", unsafe_allow_html=True)

# ============================================
#  UPLOAD GAMBAR
# ============================================
st.markdown("### 📤 Upload gambar tokoh wayang")

uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🖼 Gambar yang Diupload")

    st.image(img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Analisis Gambar"):
        # PREPROSES
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        class_id = np.argmax(predictions)
        label = class_names[class_id]

        # === HASIL ===
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-title'>✨ Hasil Prediksi: {label.title()}</div>", unsafe_allow_html=True)

        # deskripsi tambahan
        desc = wayang_info.get(label.lower(), "Deskripsi tidak tersedia.")
        st.markdown(f"### 📜 Deskripsi Tokoh\n{desc}")

        st.markdown("</div>", unsafe_allow_html=True)
