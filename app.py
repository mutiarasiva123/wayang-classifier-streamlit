import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================
#   CONFIGURATION
# ============================
st.set_page_config(
    page_title="Klasifikasi Tokoh Wayang",
    page_icon="🎭",
    layout="centered"
)

# ============================
#  CUSTOM CSS
# ============================
st.markdown("""
    <style>
        .title {
            text-align:center;
            font-size:2.4rem;
            font-weight:900;
            color:#3B1E54;
            margin-bottom:0px;
        }
        .subtitle {
            text-align:center;
            font-size:1.1rem;
            color:#6B6B6B;
            margin-top:-10px;
            margin-bottom:25px;
        }
        .result-card {
            padding:20px;
            border-radius:12px;
            background-color:#F4EFFA;
            border-left:6px solid #3B1E54;
        }
        .stButton>button {
            background-color:#3B1E54;
            color:white;
            padding:8px 25px;
            border-radius:8px;
            font-size:1rem;
            border:none;
        }
        .stButton>button:hover {
            background-color:#2A1440;
            color:white;
        }
        .footer {
            text-align:center;
            margin-top:50px;
            color:#888;
        }
    </style>
""", unsafe_allow_html=True)

# ============================
#   DESKRIPSI TOKOH WAYANG
# ============================
deskripsi_wayang = {
    "arjuna": "Arjuna adalah ksatria Pandawa yang terkenal dengan ketampanan, kebijaksanaan, dan keahlian memanahnya. Ia memiliki senjata sakti bernama Gandewa dan sering dianggap simbol kedewasaan spiritual.",
    "bagong": "Bagong adalah tokoh punakawan yang lucu, cerdas, dan sering menyampaikan kritik sosial melalui humor. Ia diciptakan oleh Semar sebagai bentuk lain dirinya.",
    "bathara surya": "Bathara Surya adalah dewa matahari dalam mitologi Jawa. Ia melambangkan cahaya, kehidupan, dan keadilan. Banyak tokoh mendapat anugerah kesaktian darinya.",
    "bathara wisnu": "Bathara Wisnu adalah dewa pemelihara alam semesta. Ia sering digambarkan sebagai dewa kebajikan, penjaga keseimbangan, dan pelindung dunia.",
    "gareng": "Gareng adalah salah satu punakawan yang memiliki sifat bijaksana, meski bentuk tubuhnya unik. Ia sering memberi nasihat filosofis namun dibungkus humor.",
    "nakula": "Nakula adalah salah satu kembar Pandawa. Ia dikenal setia, disiplin, tampan, dan sangat ahli dalam berkuda.",
    "petruk": "Petruk adalah punakawan yang berpostur tinggi, humoris, dan cerdas. Ia sering dianggap simbol rakyat kecil yang menggunakan humor untuk menyampaikan kebenaran.",
    "sadewa": "Sadewa adalah kembar dari Nakula. Ia terkenal sangat bijak, lembut hati, dan memiliki kemampuan spiritual tinggi.",
    "semar": "Semar adalah tokoh punakawan paling tua dan paling sakti. Ia adalah dewa yang turun ke bumi sebagai simbol kebijaksanaan dan pelindung kaum lemah.",
    "werkudara": "Werkudara (Bima) adalah Pandawa terkuat, berwatak tegas, jujur, dan pemberani. Ia memiliki kuku sakti Pancanaka.",
    "yudistira": "Yudistira adalah Pandawa sulung, terkenal karena kejujuran, ketenangan, dan kebijaksanaannya. Ia adalah raja Amarta."
}

# ============================
#   LOAD MODEL
# ============================
model = tf.keras.models.load_model("wayang_mobilenetv2.h5")

class_names = list(deskripsi_wayang.keys())

# ============================
#   HEADER
# ============================
st.markdown("<div class='title'>🎭 Klasifikasi Tokoh Wayang</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Unggah gambar tokoh wayang & dapatkan prediksinya lengkap dengan cerita singkat</div>", unsafe_allow_html=True)

# ============================
#   UPLOAD IMAGE
# ============================
uploaded_file = st.file_uploader("Upload gambar tokoh wayang", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Preview gambar – ukuran kecil & rapi
    st.image(image, caption="Gambar yang diupload", width=260)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Prediksi Tokoh"):
        with st.spinner("Sedang menganalisis..."):
            predictions = model.predict(img_array)
            idx = np.argmax(predictions)
            result = class_names[idx]
            confidence = predictions[0][idx] * 100

        # ============================
        #   CARD HASIL PREDIKSI
        # ============================
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.write(f"### 🎯 Tokoh Wayang: **{result.upper()}**")
        st.write(f"📊 Tingkat Kepercayaan: **{confidence:.2f}%**")
        st.write("---")
        st.write(f"### 📝 Deskripsi:")
        st.write(deskripsi_wayang[result])
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Dibuat dengan ❤️ oleh Streamlit + MobileNetV2</div>", unsafe_allow_html=True)
