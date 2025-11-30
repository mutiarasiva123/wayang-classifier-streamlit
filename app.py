import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Klasifikasi Tokoh Wayang",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# BATIK BASE64 (64×64 seamless dark batik pattern)
# ======================================================
batik_base64 = """
iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAQAAAAAYLlVAAAAnklEQVR4Ae3SsQ2AMAxF0bwB3pGc
doDdyJ3Yg7MdpBIk5+pUVdEh9jVtx2dnZ2dnZ2dnZ2eU1AObgATr0GkCUqgF6tAF6pAF6pAF6tA
F6pAF6pAF6tAF6pAF6pAF6tAF6pAF6pAF6tAF6pAF6pAF6tAF6pAF6pAF6tAF6pAF6pAF6tAF6pA
F6pAF6tAF6pAFyoCX5UW1vnh9QY4UAAAAASUVORK5CYII=
"""

# ======================================================
# GLOBAL DARK GRADIENT + BATIK BACKGROUND
# ======================================================
st.markdown(f"""
    <style>

        /* GLOBAL PAGE */
        body {{
            background-color: #0e0b16 !important;
            background-image:
                linear-gradient(rgba(0,0,0,0.72), rgba(0,0,0,0.85)),
                url("data:image/png;base64,{batik_base64}");
            background-size: 400px, 120px;
            background-attachment: fixed;
        }}

        /* MAIN CONTAINER */
        .main {{
            background: rgba(20, 18, 28, 0.80) !important;
            padding: 25px;
            border-radius: 15px;
            color: #eee !important;
        }}

        /* HEADERS */
        .hero {{
            background: linear-gradient(135deg, #2a1f49cc, #0e0b1688);
            padding: 45px;
            border-radius: 18px;
            text-align: center;
            margin-bottom: 35px;
            color: white;
        }}

        .title {{
            font-size: 3rem;
            font-weight: 900;
        }}

        .subtitle {{
            font-size: 1.3rem;
            opacity: 0.9;
        }}

        /* UPLOAD & OUTPUT CARD */
        .card {{
            background: rgba(255,255,255,0.06);
            padding: 25px;
            border-radius: 15px;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 18px rgba(0,0,0,0.45);
            color: #eee;
        }}

        /* IMAGE FRAME */
        .img-frame {{
            padding: 10px;
            border: 4px solid #7d5cd1;
            border-radius: 15px;
            background: rgba(255,255,255,0.07);
        }}

        /* BUTTON */
        .stButton>button {{
            background: linear-gradient(135deg, #8a4de8, #4d2c7a);
            border: none;
            color: white;
            padding: 12px 22px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1.05rem;
        }}

        .stButton>button:hover {{
            background: linear-gradient(135deg, #6e36c7, #371e5a);
            color: white;
        }}

        /* RESULT TEXT */
        .result-title {{
            font-size: 1.8rem;
            font-weight: 900;
            color: #d9c8ff;
            animation: fadeIn 1s ease;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

    </style>
""", unsafe_allow_html=True)


# ======================================================
# DESKRIPSI TOKOH
# ======================================================
deskripsi = {
    "arjuna": "Arjuna adalah ksatria Pandawa yang terkenal sangat tampan...",
    "bagong": "Bagong adalah punakawan yang lucu, spontan...",
    "bathara surya": "Dewa matahari pemberi terang...",
    "bathara wisnu": "Dewa pemelihara alam semesta...",
    "gareng": "Gareng adalah punakawan berhati baik...",
    "nakula": "Nakula adalah salah satu kembar Pandawa...",
    "petruk": "Petruk adalah punakawan jenaka...",
    "sadewa": "Sadewa adalah kembaran Nakula yang cerdas...",
    "semar": "Semar adalah punakawan tertua dan paling sakti...",
    "werkudara": "Werkudara/Bima adalah Pandawa terkuat...",
    "yudistira": "Yudistira adalah pemimpin Pandawa..."
}

# ======================================================
# LOAD MODEL
# ======================================================
model = tf.keras.models.load_model("wayang_mobilenetv2.h5")
class_names = list(deskripsi.keys())


# ======================================================
# HERO HEADER
# ======================================================
st.markdown("""
<div class="hero">
    <div class="title">🎭 Klasifikasi Tokoh Wayang</div>
    <div class="subtitle">Tema Gelap • Background Batik • Gradasi Elegan</div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# LAYOUT KIRI–KANAN
# ======================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload gambar wayang", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
        st.image(image, width=300)
        st.markdown("</div>", unsafe_allow_html=True)

        # Prepare image
        img = image.resize((224, 224))
        img_arr = np.expand_dims(np.array(img) / 255.0, axis=0)

        if st.button("🔍 Analisis Gambar"):
            preds = model.predict(img_arr)[0]
            top3 = np.argsort(preds)[-3:][::-1]

            st.markdown("<div class='result-title'>✨ Hasil Prediksi</div>", unsafe_allow_html=True)

            for idx in top3:
                st.write(f"**{class_names[idx].upper()}** — {preds[idx]*100:.2f}%")

            st.write("---")
            st.write("### 📝 Deskripsi Tokoh:")
            st.write(deskripsi[class_names[top3[0]]])

    st.markdown("</div>", unsafe_allow_html=True)
