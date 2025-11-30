import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================
#   CONFIG
# ============================
st.set_page_config(
    page_title="Klasifikasi Tokoh Wayang",
    page_icon="🎭",
    layout="wide",
)

# ============================
#  BACKGROUND BATIK (AMAN UNTUK STREAMLIT CLOUD)
# ============================
background_url = "https://raw.githubusercontent.com/irfnrdh/batik-assets/main/batik-pattern-cream.png"

st.markdown(
    f"""
    <style>
        body {{
            background-image: url('{background_url}');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .main {{
            background: rgba(255,255,255,0.65);
            padding: 25px;
            border-radius: 15px;
        }}

        h1 {{
            color: #000000 !important;
            font-weight: 900;
            text-shadow: 1px 1px 2px rgba(255,255,255,0.6);
        }}

        .subtitle {{
            color: #222 !important;
            font-size: 18px;
            margin-top: -10px;
        }}

        .result-card {{
            padding: 20px;
            background: rgba(255,255,255,0.85);
            border-left: 6px solid #000;
            border-radius: 12px;
        }}

        .stButton>button {{
            background-color: #000;
            color: white;
            padding: 10px 25px;
            border-radius: 8px;
            border:none;
            font-size: 17px;
        }}
        .stButton>button:hover {{
            background-color: #333;
        }}
    </style>
""",
    unsafe_allow_html=True
)

# ============================
#  DESKRIPSI WAYANG
# ============================
deskripsi_wayang = {
    "arjuna": "Arjuna adalah ksatria Pandawa yang terkenal karena ketampanan, ketenangan, dan keahlian memanah tingkat tinggi.",
    "bagong": "Bagong adalah punakawan yang lucu, kritis, dan selalu menyampaikan pesan moral melalui humor cerdas.",
    "bathara surya": "Bathara Surya adalah dewa matahari, simbol kehidupan, kekuatan, dan cahaya dalam mitologi Jawa.",
    "bathara wisnu": "Bathara Wisnu adalah dewa pemelihara alam semesta, penjaga keseimbangan dan sumber kebijaksanaan.",
    "gareng": "Gareng adalah punakawan bijak yang selalu membawa pesan filosofis melalui kelucuan dan perilakunya.",
    "nakula": "Nakula adalah ksatria Pandawa yang setia, ahli berkuda, dan memiliki watak tenang serta disiplin.",
    "petruk": "Petruk adalah punakawan berperawakan tinggi, jenaka, dan sering menjadi suara rakyat kecil yang kritis.",
    "sadewa": "Sadewa adalah kembar Nakula, terkenal bijaksana, lembut, dan memiliki kemampuan spiritual tinggi.",
    "semar": "Semar adalah punakawan tertua dan terbijak, jelmaan dewa yang menjadi pelindung manusia.",
    "werkudara": "Werkudara (Bima) adalah Pandawa terkuat, berwatak tegas, jujur, pemberani, dan berhati tulus.",
    "yudistira": "Yudistira adalah pemimpin Pandawa, simbol kejujuran, kesabaran, dan kebijaksanaan tinggi."
}

# ============================
#  LOAD MODEL
# ============================
model = tf.keras.models.load_model("wayang_mobilenetv2.h5")
class_names = list(deskripsi_wayang.keys())

# ============================
#  HEADER
# ============================
st.markdown("<h1>🎭 Klasifikasi Tokoh Wayang</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Unggah gambar tokoh wayang untuk memperoleh identifikasi lengkap dan penjelasannya.</p>", unsafe_allow_html=True)

# ============================
#  LAYOUT LANDSCAPE
# ============================
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Unggah gambar tokoh wayang", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diunggah", width=240)

        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        if st.button("🔍 Prediksi Tokoh"):
            with st.spinner("Sedang menganalisis..."):
                pred = model.predict(img_array)
                idx = np.argmax(pred)
                result = class_names[idx]
                conf = pred[0][idx] * 100

            with col2:
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.write(f"### 🎯 Tokoh: **{result.upper()}**")
                st.write(f"📊 Tingkat Keyakinan: **{conf:.2f}%**")
                st.write("---")
                st.write("### 📝 Deskripsi:")
                st.write(deskripsi_wayang[result])
                st.markdown("</div>", unsafe_allow_html=True)

# FOOTER
st.markdown("<br><center>© Dibuat dengan Streamlit + MobileNetV2</center>", unsafe_allow_html=True)
