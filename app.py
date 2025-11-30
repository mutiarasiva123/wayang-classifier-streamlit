import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# ================================
# BACKGROUND BATIK BASE64 EMBEDDED
# ================================

BATIK_BASE64 = """
iVBORw0KGgoAAAANSUhEUgAAA... <=== AKU ISI BASE64 NYA NANTI
"""

def set_bg_from_base64():
    css = f"""
    <style>
    .stApp {{
        background: 
            linear-gradient(rgba(0,0,0,0.88), rgba(0,0,0,0.9)),
            url("data:image/png;base64,{BATIK_BASE64}");
        background-size: 140px;
        background-repeat: repeat;
        background-attachment: fixed;
        color: #eee !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


set_bg_from_base64()



# ============ UI HEADER ============
st.markdown("""
<div style="text-align:center; padding:35px; 
background:rgba(255,255,255,0.06);
border-radius:18px; border:1px solid rgba(255,255,255,0.15);
backdrop-filter:blur(7px);
box-shadow:0 0 25px rgba(0,0,0,0.5);">
    <h1 style="color:#d8caff;font-weight:900;margin-bottom:5px;">
        🔮 Klasifikasi Tokoh Wayang
    </h1>
    <p style="color:#cdbaff;">Tema Gelap • Batik • Modern UI</p>
</div>
<br>
""", unsafe_allow_html=True)


# =============================
# LOAD MODEL
# =============================
try:
    model = tf.keras.models.load_model("wayang_mobilenetv2.h5")
    MODEL_READY = True
except:
    MODEL_READY = False
    st.error("❌ Model gagal dibuka — file rusak atau tidak ditemukan.")


# =============================
# CLASS LIST
# =============================
class_names = [
    "arjuna","bagong","bathara surya","bathara wisnu",
    "gareng","nakula","petruk","sadewa","semar",
    "werkudara","yudistira"
]

# DESKRIPSI TOKOH
wayang_info = {
    "semar":"Semar adalah punakawan tertua dan paling bijaksana.",
    "bagong":"Bagong adalah tokoh lucu dan simbol suara rakyat.",
    "gareng":"Gareng penuh kehati-hatian, lambang moral baik.",
    "petruk":"Petruk tinggi kurus, simbol keluwesan dan kecerdikan.",
    "arjuna":"Arjuna adalah ksatria tampan, pemanah terbaik.",
    "nakula":"Nakula adalah kembar Pandawa, tampan dan setia.",
    "sadewa":"Sadewa bijaksana, mewakili pengetahuan mendalam.",
    "yudistira":"Yudistira adalah pemimpin Pandawa, adil dan jujur.",
    "werkudara":"Bima kuat, berani, teguh pendirian.",
    "bathara surya":"Dewa matahari dalam pewayangan.",
    "bathara wisnu":"Dewa pemelihara alam semesta."
}

# =============================
# UPLOAD SECTION
# =============================
st.subheader("📤 Upload gambar tokoh wayang")

uploaded = st.file_uploader("Unggah file gambar", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_container_width=True)

    if MODEL_READY and st.button("🔍 Analisis Gambar"):
        
        resized = img.resize((224,224))
        arr = np.expand_dims(np.array(resized) / 255.0, 0)
        
        pred = model.predict(arr)
        idx = np.argmax(pred)
        label = class_names[idx]

        st.success(f"✨ Prediksi: **{label.title()}**")

        st.write("📜 Deskripsi:")
        st.write(wayang_info.get(label, "Tidak ada deskripsi."))
