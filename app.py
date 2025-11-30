import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import requests

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Klasifikasi Tokoh Wayang",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================================
# BACKGROUND BATIK (BASE64)
# =====================================================================
def set_background():
    batik_file = "batik.jpg"
    url_batik = "https://i.ibb.co/tH1G9sn/batik-purple.jpg"

    r = requests.get(url_batik)
    with open(batik_file, "wb") as f:
        f.write(r.content)

    with open(batik_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>

        body {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}

        .main {{
            background: rgba(255,255,255,0.85);
            padding: 25px;
            border-radius: 15px;
        }}

        /* HERO HEADER */
        .hero {{
            background: rgba(0,0,0,0.55);
            padding: 50px;
            text-align: center;
            border-radius: 15px;
            color: white;
            margin-bottom: 35px;
        }}

        .title {{
            font-size: 3rem;
            font-weight: 900;
        }}

        .subtitle {{
            font-size: 1.3rem;
            opacity: 0.9;
        }}

        /* CARDS */
        .card {{
            background: #ffffffee;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 4px 18px rgba(0,0,0,0.15);
        }}

        /* IMAGE FRAME */
        .img-frame {{
            padding: 10px;
            border: 5px solid #4B2E83;
            border-radius: 15px;
            background: white;
        }}

        /* RESULT TITLE */
        .result-title {{
            font-size: 1.8rem;
            font-weight: 800;
            color: #3B1E54;
            animation: fadeIn 1s ease;
        }}

        /* ANIMATION */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* MODAL POPUP */
        .modal {{
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0,0,0,0.7);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }}

        .modal-content {{
            background: white;
            padding: 30px;
            width: 60%;
            border-radius: 15px;
            animation: fadeIn 0.7s ease;
        }}

        .close-btn {{
            background: #b80000;
            color: white;
            padding: 8px 15px;
            border-radius: 10px;
            float: right;
            border: none;
        }}

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


set_background()


# =====================================================================
# WAYANG DESCRIPTIONS
# =====================================================================
deskripsi_wayang = {
    "arjuna": "Arjuna adalah ksatria Pandawa yang sangat tampan, ahli memanah, simbol kebijaksanaan.",
    "bagong": "Bagong adalah punakawan spontan dan lucu, simbol kritik sosial.",
    "bathara surya": "Dewa matahari pemberi cahaya dan kehidupan.",
    "bathara wisnu": "Dewa pemelihara alam, penuh kasih dan menjaga keseimbangan dunia.",
    "gareng": "Gareng adalah punakawan baik hati dan sederhana.",
    "nakula": "Nakula adalah Pandawa yang tampan, disiplin, dan ahli berkuda.",
    "petruk": "Petruk adalah punakawan humoris yang sering mengkritik tokoh-tokoh.",
    "sadewa": "Sadewa adalah kembar Nakula yang cerdas dan spiritualis.",
    "semar": "Semar adalah punakawan tertua, sakti, dan sesungguhnya dewa penyamar.",
    "werkudara": "Werkudara (Bima) adalah Pandawa terkuat yang jujur dan pemberani.",
    "yudistira": "Yudistira adalah pemimpin Pandawa, simbol kejujuran dan kebijaksanaan."
}


# =====================================================================
# LOAD MODEL
# =====================================================================
model = tf.keras.models.load_model("wayang_mobilenetv2.h5")
class_names = list(deskripsi_wayang.keys())


# =====================================================================
# SIDEBAR MENU
# =====================================================================
menu = st.sidebar.radio("Navigasi", ["🏠 Halaman Utama", "📘 Tentang Wayang", "👤 Tentang Pembuat"])


# =====================================================================
# PAGE 1 — KLASIFIKASI
# =====================================================================
if menu == "🏠 Halaman Utama":

    st.markdown("""
    <div class="hero">
        <div class="title">🎭 Klasifikasi Tokoh Wayang</div>
        <div class="subtitle">Unggah gambar wayang & dapatkan identitas lengkap beserta sejarahnya</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📤 Upload Gambar Tokoh", type=["jpg", "jpeg", "png"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

            st.markdown("<div class='img-frame'>", unsafe_allow_html=True)
            st.image(image, width=300)
            st.markdown("</div>", unsafe_allow_html=True)

            img_resized = image.resize((224, 224))
            img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

            if st.button("🔍 Analisis Gambar"):
                pred = model.predict(img_array)[0]
                top3_idx = np.argsort(pred)[-3:][::-1]

                st.markdown("<div class='result-title'>✨ Hasil Prediksi</div>", unsafe_allow_html=True)

                for idx in top3_idx:
                    nama = class_names[idx].upper()
                    conf = pred[idx] * 100
                    st.write(f"**{nama}** — {conf:.2f}%")

                st.write("---")

                label_top = class_names[top3_idx[0]]
                st.write("### 📝 Deskripsi Tokoh:")
                st.write(deskripsi_wayang[label_top])

        st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# PAGE 2 — TENTANG WAYANG (MODAL STYLE)
# =====================================================================
elif menu == "📘 Tentang Wayang":

    st.markdown("""
        <div class="card">
            <h2>📘 Sejarah Singkat Wayang</h2>
            Wayang adalah seni pertunjukan tradisional Indonesia yang menggabungkan seni peran, suara, musik gamelan,
            sastra, lukisan, dan filsafat. Wayang telah diakui UNESCO sebagai Mahakarya Warisan Budaya Takbenda.
            Tokoh-tokoh wayang memiliki karakter yang mencerminkan nilai kehidupan, moral, dan spiritual.
        </div>
    """, unsafe_allow_html=True)


# =====================================================================
# PAGE 3 — TENTANG PEMBUAT
# =====================================================================
elif menu == "👤 Tentang Pembuat":
    st.markdown("""
        <div class="card">
            <h2>👤 Tentang Pembuat</h2>
            <p>Aplikasi ini dibuat oleh <b>Mutiarasiva123</b> sebagai proyek Deep Learning: 
            "Klasifikasi Tokoh Wayang Menggunakan CNN & Streamlit".</p>
            <p>Aplikasi memanfaatkan MobileNetV2 untuk mengenali tokoh wayang secara akurat.</p>
        </div>
    """, unsafe_allow_html=True)
