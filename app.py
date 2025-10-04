import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="NutriScan - Deteksi Gizi Makanan",
    
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f5e9 100%);
    }
    
    /* Card Styling */
    .metric-card {
        background: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        margin: 12px 0;
        border-left: 4px solid #4CAF50;
    }
    
    /* Header Styling */
    h1 {
        color: #2e7d32;
        font-weight: 700;
        margin-bottom: 8px;
        font-size: 2.5rem;
    }
    
    h2 {
        color: #388e3c;
        font-weight: 600;
        font-size: 1.5rem;
    }
    
    h3 {
        color: #43a047;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(76, 175, 80, 0.2);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #388e3c 0%, #4CAF50 100%);
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
        transform: translateY(-1px);
    }
    
    /* Upload Box */
    .upload-box {
        background: white;
        padding: 48px;
        border-radius: 12px;
        border: 2px dashed #81c784;
        text-align: center;
        margin: 20px 0;
    }
    
    /* Result Box */
    .result-box {
        background: white;
        padding: 28px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin: 20px 0;
        border-top: 3px solid #4CAF50;
    }
    
    /* Category Badge */
    .category-badge {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric Styling */
    .stMetric {
        background: #f1f8f4;
        padding: 12px;
        border-radius: 8px;
        border-left: 3px solid #4CAF50;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2e7d32 0%, #388e3c 100%);
    }
    
    .css-1d391kg .stRadio label, [data-testid="stSidebar"] .stRadio label {
        color: white !important;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def create_food_dataset():
    foods = {
        'nama_makanan': [
            'Nasi Putih', 'Nasi Goreng', 'Mie Goreng', 'Ayam Goreng', 'Rendang',
            'Soto Ayam', 'Gado-gado', 'Sayur Asem', 'Tempe Goreng', 'Tahu Goreng',
            'Ikan Bakar', 'Pecel Lele', 'Sate Ayam', 'Bakso', 'Nasi Uduk',
            'Capcay', 'Tumis Kangkung', 'Pepes Ikan', 'Telur Dadar', 'Bubur Ayam',
            'Sop Buah', 'Salad Sayur', 'Nasi Kuning', 'Ayam Bakar', 'Rawon',
            'Sate Padang', 'Ketoprak', 'Pecel', 'Nasi Liwet', 'Sup Iga'
        ],
        'kalori': [180, 350, 420, 290, 380, 150, 250, 80, 150, 140, 220, 280, 200, 180, 300,
                   120, 90, 180, 160, 140, 95, 65, 310, 270, 320, 210, 280, 230, 290, 260],
        'protein': [4, 12, 15, 28, 25, 18, 8, 3, 14, 10, 25, 20, 18, 10, 8,
                    6, 4, 22, 12, 8, 2, 3, 9, 26, 20, 19, 10, 12, 7, 22],
        'lemak': [0.5, 15, 18, 15, 22, 5, 12, 2, 8, 7, 10, 15, 8, 6, 12,
                  4, 3, 8, 12, 4, 0.3, 2, 13, 14, 16, 9, 11, 10, 11, 12],
        'karbohidrat': [40, 45, 55, 10, 8, 15, 25, 12, 8, 10, 5, 20, 10, 25, 45,
                        15, 10, 5, 2, 22, 20, 8, 48, 9, 18, 12, 35, 28, 46, 15],
        'serat': [0.5, 2, 3, 0, 1, 2, 6, 4, 3, 2, 0, 1, 0.5, 1, 1.5,
                  4, 3, 1, 0, 1, 3.5, 5, 2, 0.5, 2, 1, 5, 6, 1.8, 2],
        # Fitur visual yang lebih detail
        'red_avg': [245, 210, 200, 180, 140, 230, 160, 190, 150, 220, 160, 140, 150, 240, 235,
                    180, 120, 170, 250, 245, 255, 145, 238, 175, 95, 155, 195, 165, 242, 185],
        'green_avg': [240, 180, 160, 140, 100, 220, 140, 180, 120, 200, 130, 110, 120, 230, 220,
                      170, 140, 140, 240, 235, 250, 180, 225, 135, 75, 125, 175, 155, 232, 155],
        'blue_avg': [235, 130, 120, 100, 70, 200, 100, 150, 80, 160, 100, 80, 90, 210, 200,
                     130, 100, 110, 200, 220, 245, 155, 195, 95, 60, 95, 145, 130, 210, 130],
        'brightness': [240, 173, 160, 140, 103, 217, 133, 173, 117, 193, 130, 110, 120, 227, 218,
                       160, 120, 140, 230, 233, 250, 160, 219, 135, 77, 125, 172, 150, 228, 157],
        'saturation': [0.04, 0.38, 0.40, 0.44, 0.50, 0.13, 0.45, 0.27, 0.47, 0.27, 0.38, 0.43, 0.40, 0.13, 0.15,
                       0.28, 0.33, 0.35, 0.22, 0.10, 0.02, 0.24, 0.18, 0.46, 0.37, 0.39, 0.26, 0.27, 0.14, 0.35]
    }
    
    df = pd.DataFrame(foods)
    
    # Klasifikasi kategori gizi yang lebih akurat
    def classify_nutrition(row):
        if row['kalori'] < 100 and row['lemak'] < 3:
            return 'Sangat Rendah Kalori'
        elif row['kalori'] < 150 and row['lemak'] < 5:
            return 'Rendah Kalori'
        elif row['protein'] > 20:
            return 'Tinggi Protein'
        elif row['lemak'] > 15:
            return 'Tinggi Lemak'
        elif row['karbohidrat'] > 40:
            return 'Tinggi Karbohidrat'
        elif row['serat'] > 4:
            return 'Tinggi Serat'
        else:
            return 'Seimbang'
    
    df['kategori_gizi'] = df.apply(classify_nutrition, axis=1)
    return df

# Fungsi ekstraksi fitur yang lebih lengkap
def extract_color_features(image):
    # Resize untuk processing
    img = image.resize((150, 150))
    img_array = np.array(img)
    
    # Pastikan RGB (3 channel)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # Hitung rata-rata RGB
    red_avg = np.mean(img_array[:, :, 0])
    green_avg = np.mean(img_array[:, :, 1])
    blue_avg = np.mean(img_array[:, :, 2])
    
    # Hitung brightness
    brightness = np.mean(img_array)
    
    # Hitung saturation (simplified)
    max_rgb = np.max(img_array, axis=2)
    min_rgb = np.min(img_array, axis=2)
    saturation = np.mean((max_rgb - min_rgb) / (max_rgb + 1e-6))
    
    return red_avg, green_avg, blue_avg, brightness, saturation

# Fungsi training model yang lebih baik
@st.cache_resource
def train_models(df):
    X = df[['red_avg', 'green_avg', 'blue_avg', 'brightness', 'saturation']]
    
    # Standardisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model untuk masing-masing nutrisi dengan parameter yang lebih baik
    model_kalori = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    model_protein = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    model_lemak = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    model_karbohidrat = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    model_serat = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    
    model_kalori.fit(X_scaled, df['kalori'])
    model_protein.fit(X_scaled, df['protein'])
    model_lemak.fit(X_scaled, df['lemak'])
    model_karbohidrat.fit(X_scaled, df['karbohidrat'])
    model_serat.fit(X_scaled, df['serat'])
    
    return {
        'kalori': model_kalori,
        'protein': model_protein,
        'lemak': model_lemak,
        'karbohidrat': model_karbohidrat,
        'serat': model_serat,
        'scaler': scaler
    }

# Fungsi prediksi yang lebih akurat
def predict_nutrition(image, models):
    red, green, blue, brightness, saturation = extract_color_features(image)
    features = np.array([[red, green, blue, brightness, saturation]])
    features_scaled = models['scaler'].transform(features)
    
    predictions = {
        'kalori': round(max(50, min(500, models['kalori'].predict(features_scaled)[0])), 1),
        'protein': round(max(0, min(40, models['protein'].predict(features_scaled)[0])), 1),
        'lemak': round(max(0, min(30, models['lemak'].predict(features_scaled)[0])), 1),
        'karbohidrat': round(max(0, min(70, models['karbohidrat'].predict(features_scaled)[0])), 1),
        'serat': round(max(0, min(10, models['serat'].predict(features_scaled)[0])), 1)
    }
    
    # Klasifikasi kategori yang lebih detail
    if predictions['kalori'] < 100 and predictions['lemak'] < 3:
        kategori = 'Sangat Rendah Kalori'
        color = '#81c784'
    elif predictions['kalori'] < 150 and predictions['lemak'] < 5:
        kategori = 'Rendah Kalori'
        color = '#66bb6a'
    elif predictions['protein'] > 20:
        kategori = 'Tinggi Protein'
        color = '#ff9800'
    elif predictions['lemak'] > 15:
        kategori = 'Tinggi Lemak'
        color = '#ef5350'
    elif predictions['karbohidrat'] > 40:
        kategori = 'Tinggi Karbohidrat'
        color = '#42a5f5'
    elif predictions['serat'] > 4:
        kategori = 'Tinggi Serat'
        color = '#26a69a'
    else:
        kategori = 'Seimbang'
        color = '#ab47bc'
    
    # Hitung confidence score sederhana
    confidence = min(95, 70 + (saturation * 30))
    
    return predictions, kategori, color, confidence

# Load data dan model
df_foods = create_food_dataset()
models = train_models(df_foods)

# Header dengan icon
col_header1, col_header2 = st.columns([0.1, 0.9])
with col_header1:
    st.markdown("<h1 style='font-size: 3rem; margin: 0;'>🥗</h1>", unsafe_allow_html=True)
with col_header2:
    st.title("NutriScan")
    st.markdown("<p style='color: #558b2f; font-size: 1.1rem; margin-top: -10px;'>Analisis Kandungan Gizi Makanan dengan AI</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: white; text-align: center;'>📋 Menu</h2>", unsafe_allow_html=True)
    menu = st.radio("", ["🔍 Deteksi Gizi", "📊 Database Makanan", "ℹ️ Tentang Sistem"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("""
        <div style='color: white; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 8px;'>
            <h4 style='color: white;'>💡 Tips</h4>
            <p style='font-size: 0.9rem;'>Foto dengan pencahayaan baik dan fokus pada makanan untuk hasil terbaik</p>
        </div>
    """, unsafe_allow_html=True)

# Menu Deteksi Gizi
if menu == "🔍 Deteksi Gizi":
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 📸 Upload Gambar Makanan")
        
        uploaded_file = st.file_uploader(
            "Pilih gambar makanan (JPG, PNG, JPEG)",
            type=['jpg', 'jpeg', 'png'],
            help="Unggah foto makanan dengan pencahayaan yang baik untuk hasil terbaik",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Gambar yang diunggah", use_container_width=True)
            
            if st.button("🔬 Analisis Kandungan Gizi"):
                progress_bar = st.progress(0)
                with st.spinner("Menganalisis gambar..."):
                    for i in range(100):
                        progress_bar.progress(i + 1)
                    
                    predictions, kategori, color, confidence = predict_nutrition(image, models)
                    st.session_state['predictions'] = predictions
                    st.session_state['kategori'] = kategori
                    st.session_state['color'] = color
                    st.session_state['confidence'] = confidence
                    
                st.success("✅ Analisis selesai!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            kategori = st.session_state['kategori']
            color = st.session_state['color']
            confidence = st.session_state['confidence']
            
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown("### 📊 Hasil Analisis")
            
            # Confidence score
            st.markdown(f"""
                <div style='background: #f1f8f4; padding: 12px; border-radius: 8px; margin-bottom: 20px;'>
                    <p style='margin: 0; color: #2e7d32;'><strong>Tingkat Kepercayaan:</strong> {confidence:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Kategori
            st.markdown(f"""
                <div style='text-align: center; margin: 24px 0;'>
                    <p style='color: #616161; margin-bottom: 12px; font-weight: 500;'>Kategori Gizi</p>
                    <span class='category-badge' style='background-color: {color}; color: white;'>
                        {kategori}
                    </span>
                </div>
            """, unsafe_allow_html=True)
            
            # Tabel nutrisi
            st.markdown("#### 🍱 Kandungan Nutrisi (per 100g)")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("⚡ Kalori", f"{predictions['kalori']:.1f} kcal")
                st.metric("🥩 Protein", f"{predictions['protein']:.1f} g")
                st.metric("🧈 Lemak", f"{predictions['lemak']:.1f} g")
            
            with col_b:
                st.metric("🍚 Karbohidrat", f"{predictions['karbohidrat']:.1f} g")
                st.metric("🌾 Serat", f"{predictions['serat']:.1f} g")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Visualisasi
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("#### 📈 Visualisasi Nutrisi")
            
            nutrisi_data = pd.DataFrame({
                'Nutrisi': ['Protein', 'Lemak', 'Karbohidrat', 'Serat'],
                'Jumlah (g)': [
                    predictions['protein'],
                    predictions['lemak'],
                    predictions['karbohidrat'],
                    predictions['serat']
                ],
                'Warna': ['#4CAF50', '#FF9800', '#2196F3', '#9C27B0']
            })
            
            fig = go.Figure(data=[
                go.Bar(
                    x=nutrisi_data['Nutrisi'],
                    y=nutrisi_data['Jumlah (g)'],
                    marker_color=nutrisi_data['Warna'],
                    text=nutrisi_data['Jumlah (g)'].round(1),
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                height=320,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Rekomendasi
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("#### 💡 Rekomendasi Konsumsi")
            
            if kategori == "Tinggi Protein":
                st.info("✅ Sangat baik untuk pembentukan otot dan pemulihan. Ideal dikonsumsi setelah olahraga atau sebagai menu utama.")
            elif kategori == "Tinggi Lemak":
                st.warning("⚠️ Konsumsi dalam jumlah sedang. Perhatikan porsi dan kombinasikan dengan sayuran untuk keseimbangan nutrisi.")
            elif kategori == "Tinggi Karbohidrat":
                st.info("⚡ Sumber energi utama. Baik untuk aktivitas fisik tinggi. Seimbangkan dengan protein dan sayuran.")
            elif kategori == "Rendah Kalori" or kategori == "Sangat Rendah Kalori":
                st.success("✅ Cocok untuk diet penurunan berat badan. Dapat dikonsumsi dalam porsi lebih besar.")
            elif kategori == "Tinggi Serat":
                st.success("✅ Sangat baik untuk pencernaan. Perbanyak minum air putih untuk hasil optimal.")
            else:
                st.success("✅ Nutrisi seimbang untuk konsumsi sehari-hari. Baik untuk menu regular.")
            
            # Info tambahan
            total_makro = predictions['protein'] + predictions['lemak'] + predictions['karbohidrat']
            st.markdown(f"""
                <div style='background: #f5f5f5; padding: 16px; border-radius: 8px; margin-top: 16px;'>
                    <p style='margin: 0; color: #424242; font-size: 0.9rem;'>
                        <strong>📌 Catatan:</strong> Estimasi berdasarkan analisis visual dengan tingkat kepercayaan {confidence:.1f}%. 
                        Total makronutrien: {total_makro:.1f}g
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# Menu Database Makanan
elif menu == "📊 Database Makanan":
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 📚 Database Makanan Indonesia")
    
    # Filter
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        kategori_filter = st.multiselect(
            "🏷️ Filter Kategori Gizi:",
            options=sorted(df_foods['kategori_gizi'].unique()),
            default=sorted(df_foods['kategori_gizi'].unique())
        )
    
    with col_f2:
        kalori_range = st.slider(
            "⚡ Range Kalori:",
            int(df_foods['kalori'].min()),
            int(df_foods['kalori'].max()),
            (int(df_foods['kalori'].min()), int(df_foods['kalori'].max()))
        )
    
    df_filtered = df_foods[
        (df_foods['kategori_gizi'].isin(kategori_filter)) &
        (df_foods['kalori'] >= kalori_range[0]) &
        (df_foods['kalori'] <= kalori_range[1])
    ]
    
    # Display data
    display_df = df_filtered[['nama_makanan', 'kalori', 'protein', 'lemak', 
                               'karbohidrat', 'serat', 'kategori_gizi']].copy()
    display_df.columns = ['Nama Makanan', 'Kalori (kcal)', 'Protein (g)', 
                          'Lemak (g)', 'Karbohidrat (g)', 'Serat (g)', 'Kategori']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Statistik
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("📋 Total Makanan", len(df_filtered))
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("⚡ Avg Kalori", f"{df_filtered['kalori'].mean():.0f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("🥩 Avg Protein", f"{df_filtered['protein'].mean():.1f}g")
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("🏷️ Kategori", df_filtered['kategori_gizi'].nunique())
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Grafik distribusi
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Distribusi Kategori Gizi")
        fig_pie = px.pie(df_filtered, names='kategori_gizi', hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_pie.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_chart2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### 📈 Top 10 Makanan Berkalori Tinggi")
        top_10 = df_filtered.nlargest(10, 'kalori')[['nama_makanan', 'kalori']]
        fig_bar = px.bar(top_10, x='kalori', y='nama_makanan', orientation='h',
                        color='kalori', color_continuous_scale='Greens')
        fig_bar.update_layout(height=350, showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)