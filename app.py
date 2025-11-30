import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# ============================
#   CONFIG PAGE
# ============================
st.set_page_config(
    page_title="Klasifikasi Tokoh Wayang",
    page_icon="🎭",
    layout="centered"
)

# ============================
#   LOAD BACKGROUND IMAGE
# ============================
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        body {{
            background: linear-gradient(135deg, #1a1a1a, #2d0e37);
            background-attachment: fixed;
        }}
        .main {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: 450px;
            background-repeat: repeat;
            background-blend-mode: soft-light;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("batik.png")  # file batik kamu


# ============================
#  CUSTOM CSS
# ============================
st.markdown("""
<style>
/* ================= HEADER ================= */
.header-box {
    text-align:center;
    padding:20px 10px;
}
.title {
    font-size:2.7rem;
    font-weight:900;
    color:#E8D5FF;
    text-shadow:0 0 12px #5900b3;
}
.subtitle {
    font-size:1.1rem;
    color:#C9C9C9;
    margin-top:-8px;
}

/* ================= UPLOADER ================= */
.upload-section {
    background-color:rgba(255,255,255,0.08);
    padding:18px;
    margin-top:20px;
    border-radius:12px;
    border:1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(6px);
}

/* ================= BUTTON ================= */
.stButton>button {
    background: linear-gradient(135deg, #6f2dbd, #3b1e54);
    color:white;
    padding:10px 28px;
    border-radius:9px;
    border:none;
    font-size:1.05rem;
    font-weight:600;
    letter-spacing:0.5px;
    box-shadow:0 0 8px #4e1a7e;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #531b8f, #2a1440);
    box-shadow:0 0 12px #b37bff;
}

/* ================= IMAGE PREVIEW ================= */
.preview-img {
    text-align:center;
    margin-top:20px;
}

/* ================= RESULT CARD ================= */
.result-card {
    margin-top:25px;
    padding:22px;
    border-radius:14px;
    background-color:rgba(255,255,255,0.08);
    border-left:6px solid #b37bff;
    backdrop-filter: blur(6px);
    color:#F0E9FF;
}

/* ================= FOOTER ================= */
.footer {
    margin-top:55px;
    text-align:center;
    color:#bfbfbf;
    font-size:0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ============================
#   DESKRIPSI WAYANG
# ============================
deskripsi_wayang = {
    "arjuna": "Arjuna merupakan ksatria Pandawa yang tersohor karena ketampanan, kebijaksanaan, serta kemahirannya dalam ilmu memanah.",
    "bagong": "Bagong adalah punakawan yang cerdas dan humoris. Ia sering menyampaikan kritik sosial melalui candaan.",
    "bathara surya": "Bathara Surya adalah dewa matahari yang menjadi simbol kehidupan, cahaya, dan keadilan dalam mitologi Jawa.",
    "bathara wisnu": "Bathara Wisnu merupakan dewa pemelihara yang menjaga keseimbangan alam dan menjadi pelindung dunia.",
    "gareng": "Gareng adalah punakawan berhati bijak, meskipun bentuk tubuhnya unik. Ia kerap memberi nasihat dengan sentuhan humor.",
    "nakula": "Nakula adalah saudara kembar Sadewa, terkenal setia, disiplin, serta mahir dalam berkuda.",
    "petruk": "Petruk merupakan punakawan berpostur tinggi yang jenaka dan bijaksana. Ia sering menjadi simbol suara kebenaran rakyat kecil.",
    "sadewa": "Sadewa adalah kembar Nakula, terkenal berwatak lembut, cerdas, dan memiliki kemampuan spiritual tinggi.",
    "semar": "Semar adalah punakawan tertua sekaligus paling sakti. Ia melambangkan kebijaksanaan dan penjaga kaum lemah.",
    "werkudara": "Werkudara atau Bima adalah Pandawa yang kuat, jujur, serta memiliki kuku sakti Pancanaka.",
    "yudistira": "Yudistira merupakan Pandawa sulung yang dikenal akan kejujuran, kesabaran, dan kebijaksanaannya."
}

# ============================
#   LOAD MODEL
# ============================
model = tf.keras.models.load_model("wayang_mobilenetv2.h5")
class_names = list(deskripsi_wayang.keys())


# ============================
#   HEADER
# ============================
st.markdown("""
<div class='header-box'>
    <div class='title'>🎭 Klasifikasi Tokoh Wayang</div>
    <div class='subtitle'>Unggah gambar tokoh wayang untuk memperoleh hasil identifikasi dan penjelasan lengkap</div>
</div>
""", unsafe_allow_html=True)


# ============================
#   UPLOAD IMAGE
# ============================
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Unggah gambar tokoh wayang", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # PREVIEW GAMBAR KECIL & RAPI
    st.markdown("<div class='preview-img'>", unsafe_allow_html=True)
    st.image(image, caption="Pratinjau Gambar", width=240)
    st.markdown("</div>", unsafe_allow_html=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Analisis Gambar"):
        with st.spinner("Sedang memproses..."):
            predictions = model.predict(img_array)
            idx = np.argmax(predictions)
            result = class_names[idx]
            confidence = predictions[0][idx] * 100

        # ============================
        #   SHOW RESULT CARD
        # ============================
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.write(f"### 🎯 Tokoh: **{result.upper()}**")
        st.write(f"📊 Tingkat Kepercayaan: **{confidence:.2f}%**")
        st.write("---")
        st.write("### 📝 Deskripsi Tokoh")
        st.write(deskripsi_wayang[result])
        st.markdown("</div>", unsafe_allow_html=True)

# ============================
#   FOOTER
# ============================
st.markdown("<div class='footer'>Dibuat dengan ❤️ menggunakan Streamlit & MobileNetV2</div>", unsafe_allow_html=True)
