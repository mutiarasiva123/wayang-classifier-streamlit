import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Klasifikasi Tokoh Wayang",
    page_icon="🎭",
    layout="wide"
)

# ============================
# FUNGSI BACKGROUND
# ============================
def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        body {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }}
        .main {{
            background: rgba(0,0,0,0.55);
            padding: 30px;
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# PASANG BACKGROUND
set_background("batik.png")

# ============================
# DESKRIPSI TOKOH WAYANG
# ============================
deskripsi_wayang = {
    "arjuna": "Arjuna merupakan ksatria Pandawa yang terkenal dengan ketampanan, kebijaksanaan, serta keterampilannya dalam memanah.",
    "bagong": "Bagong adalah punakawan berkarakter jenaka dan kritis. Ia sering menyampaikan pesan moral melalui humor.",
    "bathara surya": "Bathara Surya adalah dewa matahari yang melambangkan sumber kehidupan dan keadilan.",
    "bathara wisnu": "Bathara Wisnu adalah dewa pemelihara yang menjaga keseimbangan dunia.",
    "gareng": "Gareng adalah punakawan yang bijak, berperilaku hati-hati, dan sering mengajarkan nilai moral.",
    "nakula": "Nakula adalah salah satu kembar Pandawa yang terkenal disiplin, setia, dan ahli berkuda.",
    "petruk": "Petruk adalah punakawan berpostur tinggi dengan sifat humoris dan cerdas.",
    "sadewa": "Sadewa merupakan saudara kembar Nakula yang bijaksana serta memiliki kemampuan spiritual tinggi.",
    "semar": "Semar adalah punakawan tertua dan paling dihormati. Ia melambangkan kebijaksanaan dan kerendahan hati.",
    "werkudara": "Werkudara (Bima) merupakan Pandawa terkuat, berwatak tegas dan jujur.",
    "yudistira": "Yudistira adalah Pandawa sulung yang terkenal karena kejujurannya dan ketenangannya."
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
    <h1 style="
        color:#E8D5FF;
        font-weight:900;
        text-shadow:0px 0px 6px #c084fc;
        margin-bottom:0;
    ">🎭 Klasifikasi Tokoh Wayang</h1>
    <p style="color:#f0e9ff; margin-top:-8px;">
        Unggah gambar tokoh wayang untuk memperoleh hasil identifikasi dan penjelasan lengkap.
    </p>
""", unsafe_allow_html=True)

st.write("")  # spasi

# ============================
# LAYOUT LANDSCAPE
# ============================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("<h4 style='color:#fff;'>Unggah Gambar</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        # Tampilkan preview kecil
        st.image(image, caption="Gambar terunggah", width=220)

        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predict_btn = st.button("🔍 Analisis Gambar")

with col2:
    if uploaded_file and predict_btn:
        with st.spinner("Sedang memproses gambar..."):
            predictions = model.predict(img_array)
            idx = np.argmax(predictions)
            result = class_names[idx]
            confidence = predictions[0][idx] * 100

        # Kartu hasil
        st.markdown("""
            <div style="
                background: rgba(255,255,255,0.15);
                padding:20px;
                border-radius:12px;
                backdrop-filter: blur(8px);
                color:white;
            ">
        """, unsafe_allow_html=True)

        st.markdown(f"## 🎯 Tokoh: **{result.upper()}**")
        st.markdown(f"### 📊 Tingkat Kepercayaan: **{confidence:.2f}%**")
        st.markdown("---")
        st.markdown("### 📝 Deskripsi Tokoh:")
        st.markdown(f"<p style='font-size:16px;'>{deskripsi_wayang[result]}</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# FOOTER
st.markdown("""
    <p style='text-align:center; color:#eee; margin-top:40px;'>
        Dibangun dengan Streamlit & MobileNetV2 • © 2025
    </p>
""", unsafe_allow_html=True)
