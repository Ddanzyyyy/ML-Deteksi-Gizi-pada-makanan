import streamlit as st
from PIL import Image
import numpy as np
import json
from ultralytics import YOLO

st.set_page_config(
    page_title="Deteksi Gizi Makanan dengan YOLO",
    layout="centered"
)

st.markdown("""
<style>
body, .stApp {
    background: #f9f9f9;
    font-family: 'Inter', sans-serif;
}
.stButton > button {
    background-color: #2d6a4f;
    color: white;
    font-weight: 500;
    border-radius: 6px;
    padding: 8px 24px;
    border: none;
}
.stButton > button:hover {
    background-color: #40916c;
}
.stFileUploader, .stTextInput {
    background: #fff;
    border-radius: 8px;
}
.stDataFrame, .stTable {
    border-radius: 8px;
    background: #fff;
}
.result-box {
    background: #fff;
    border-radius: 10px;
    padding: 28px 18px;
    margin-top: 18px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.07);
    border-top: 3px solid #2d6a4f;
}
.header-title {
    font-size: 2.1rem;
    font-weight: 600;
    color: #2d6a4f;
    margin-bottom: 2px;
}
.header-desc {
    color: #476d5a;
    font-size: 1.08rem;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header-title'>Deteksi Gizi Makanan</div>", unsafe_allow_html=True)
st.markdown("<div class='header-desc'>Deteksi jenis makanan dari gambar dan estimasi kandungan gizi per 100g menggunakan YOLOv8 dan database gizi.</div>", unsafe_allow_html=True)

# --- LOAD YOLO MODEL ---
@st.cache_resource
def load_model():
    # Pastikan path ke model benar
    return YOLO('best.pt')

model = load_model()

# --- LOAD NUTRITION DATABASE ---
@st.cache_data
def load_nutrition_db():
    with open('nutrition_db.json', 'r') as f:
        db = json.load(f)
    return db

nutrition_db = load_nutrition_db()

# --- MAIN UI ---
uploaded_file = st.file_uploader("Upload gambar makanan (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    
    if st.button("Deteksi & Estimasi Gizi"):
        with st.spinner("Deteksi makanan & estimasi gizi..."):
            # YOLOv8 prediction
            results = model.predict(np.array(image), conf=0.4)
            result = results[0]
            
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.subheader("Hasil Deteksi YOLO")
            
            if len(result.boxes) == 0:
                st.warning("Tidak ada makanan terdeteksi pada gambar.")
            else:
                # Tabel hasil deteksi
                detected_foods = []
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])
                    detected_foods.append(label)
                
                detected_unique = list(dict.fromkeys(detected_foods))
                st.write(f"Jumlah makanan terdeteksi: **{len(detected_foods)}**")
                
                det_df = []
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])
                    det_df.append({'Makanan': label, 'Kepercayaan': f"{conf:.2f}"})
                
                st.table(det_df)
            
                st.subheader("Estimasi Kandungan Gizi per 100gram")
                nutri_rows = []
                for label in detected_unique:
                    if label in nutrition_db:
                        nut = nutrition_db[label]
                        nutri_rows.append({
                            'Makanan': label,
                            'Kalori (kcal)': nut['kalori'],
                            'Protein (g)': nut['protein'],
                            'Lemak (g)': nut['lemak'],
                            'Karbohidrat (g)': nut['karbohidrat'],
                        })
                    else:
                        nutri_rows.append({
                            'Makanan': label,
                            'Kalori (kcal)': '-',
                            'Protein (g)': '-',
                            'Lemak (g)': '-',
                            'Karbohidrat (g)': '-',
                        })
                st.dataframe(nutri_rows, use_container_width=True, hide_index=True)
                
                st.info("Estimasi gizi berdasarkan database sederhana. Untuk hasil lebih akurat, gunakan food recognition dataset yang lebih komprehensif.")
            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload gambar makanan untuk mendeteksi dan estimasi gizi.")

with st.expander("Lihat Database Kandungan Gizi Makanan"):
    db_table = []
    for food, nut in nutrition_db.items():
        db_table.append({
            'Makanan': food,
            'Kalori (kcal)': nut['kalori'],
            'Protein (g)': nut['protein'],
            'Lemak (g)': nut['lemak'],
            'Karbohidrat (g)': nut['karbohidrat'],
        })
    st.dataframe(db_table, use_container_width=True, hide_index=True)

st.caption("© Deteksi Gizi Makanan dengan YOLOv8 & Streamlit | UI minimalis")
