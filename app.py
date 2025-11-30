st.markdown(f"""
<style>

    /* WRAPPER UTAMA STREAMLIT */
    .stApp {{
        background: 
            linear-gradient(rgba(0,0,0,0.80), rgba(0,0,0,0.85)),
            url("data:image/png;base64,{batik_base64}");
        background-size: 180px;
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
        background: rgba(15, 11, 25, 0.55);
        border-radius: 20px;
        box-shadow: 0 0 30px rgba(0,0,0,0.45);
        backdrop-filter: blur(6px);
    }}

    /* HEADER CARD */
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
    }}

    /* CARD UMUM */
    .card {{
        background: rgba(255,255,255,0.08);
        padding: 22px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.08);
        color: #ddd;
        backdrop-filter: blur(8px);
    }}

    /* UPLOADER BOX (versi baru Streamlit) */
    [data-testid="stFileUploader"] {{
        background: rgba(255,255,255,0.07);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.12);
        color: #ddd;
    }}

    /* IMAGE FRAME */
    .img-frame {{
        padding: 10px;
        border: 4px solid #7d5cd1;
        border-radius: 15px;
        background: rgba(255,255,255,0.05);
    }}

    /* MODERN BUTTON */
    .stButton > button {{
        background: linear-gradient(135deg, #8a4de8, #4d2c7a);
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1.05rem;
        transition: 0.2s;
        box-shadow: 0 0 15px rgba(138,77,232,0.4);
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, #6d37c5, #341f59);
        transform: scale(1.03);
        box-shadow: 0 0 25px rgba(138,77,232,0.7);
    }}

    /* JUDUL HASIL */
    .result-title {{
        font-size: 1.7rem;
        font-weight: 800;
        color: #d8caff;
        padding-top: 10px;
        animation: fadeIn 1s ease;
    }}

    /* ANIMASI */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

</style>
""", unsafe_allow_html=True)
