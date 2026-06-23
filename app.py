import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="SIAGA LAUT - Monitoring Illegal Fishing Sultra",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8fafc;
    }
    
    .stButton>button {
        width: 100%;
        background-color: #1e40af;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.2);
    }
    
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.1);
        border-left: 5px solid #2563eb;
        text-align: center;
    }
    
    .metric-val {
        font-size: 24px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 5px;
    }
    
    .metric-lbl {
        font-size: 13px;
        color: #64748b;
        font-weight: 500;
    }
    
    .result-card {
        padding: 25px;
        border-radius: 12px;
        margin-top: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .res-high {
        background-color: #fef2f2;
        border: 1px solid #fee2e2;
        border-left: 6px solid #ef4444;
    }
    
    .res-medium {
        background-color: #fffbeb;
        border: 1px solid #fef3c7;
        border-left: 6px solid #f59e0b;
    }
    
    .res-low {
        background-color: #f0fdf4;
        border: 1px solid #dcfce7;
        border-left: 6px solid #22c55e;
    }

    [data-testid="stSidebar"] .stRadio i {
        font-size: 16px;
    }
    [data-testid="stSidebar"] .stRadio label p {
        font-size: 15px !important;
        font-weight: 500 !important;
        color: #f1f5f9 !important;
        padding-top: 2px;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > div {
        background-color: rgba(255, 255, 255, 0.04);
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > div:hover {
        background-color: rgba(255, 255, 255, 0.08);
        border-color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "random_forest_model_terbaik.joblib"
MODEL_COLUMNS = [
    "Fishing_Hours_Decimal",
    "Total_Trip_Duration_Hours",
    "Fishing_Efficiency_Ratio",
    "Month",
    "Day_of_Week",
    "Is_Weekend",
    "Entry_Session"
]

@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        return None

model = load_model(MODEL_PATH)

if 'history' not in st.session_state:
    st.session_state.history = []

if 'reset_trigger' not in st.session_state:
    st.session_state.reset_trigger = False

def get_cluster_details(cluster_id):
    if cluster_id == 0:
        return {
            "name": "Mencurigakan / Potensi Illegal Fishing",
            "risk": "Risiko Tinggi (🔴)",
            "desc": "Pola aktivitas kapal menunjukkan karakteristik yang menyimpang dan memiliki potensi terindikasi melakukan Illegal Fishing sehingga memerlukan perhatian dan pengawasan lebih lanjut.",
            "class_css": "res-high",
            "color": "#ef4444"
        }
    elif cluster_id == 1:
        return {
            "name": "Normal / Sedang",
            "risk": "Risiko Sedang (🟡)",
            "desc": "Pola aktivitas kapal masih berada dalam kategori normal, namun tetap memerlukan monitoring secara berkala.",
            "class_css": "res-medium",
            "color": "#f59e0b"
        }
    else:
        return {
            "name": "Aman / Rendah",
            "risk": "Risiko Rendah (🟢)",
            "desc": "Pola aktivitas kapal berada pada kondisi yang stabil dan tidak menunjukkan indikasi aktivitas Illegal Fishing.",
            "class_css": "res-low",
            "color": "#22c55e"
        }

st.markdown("""
<div style="background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); padding: 30px; border-radius: 16px; color: white; margin-bottom: 30px; box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.3);">
    <h1 style='margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -0.5px;'>🚢 Klasifikasi Pola Aktivitas Kapal Perikanan Berdasarkan Data AIS untuk Identifikasi Potensi Illegal Fishing di Perairan Sulawesi Tenggara</h1>
    <p style='margin: 8px 0 0 0; font-size: 15px; opacity: 0.9; font-weight: 400;'>Sistem Prediksi dan Monitoring Aktivitas Kapal Perikanan Berbasis Machine Learning</p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-card"><div class="metric-val">5.737 Data</div><div class="metric-lbl">📊 JUMLAH DATASET</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><div class="metric-val">99,13%</div><div class="metric-lbl">🎯 AKURASI MODEL</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><div class="metric-val">80%</div><div class="metric-lbl">🧠 DATA TRAINING</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-card"><div class="metric-val">20%</div><div class="metric-lbl">🧪 DATA TESTING</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 20px 0; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 25px;">
    <span style="font-size: 50px; display: block; margin-bottom: 5px;">🌊</span>
    <h2 style="margin: 0; color: #3b82f6; font-size: 22px; font-weight: 800; letter-spacing: 0.5px;">SIAGA LAUT</h2>
    <p style="margin: 5px 0 0 0; color: #94a3b8; font-size: 11px; line-height: 1.4; text-transform: uppercase; font-weight: 600;">
        Sistem Integrasi Analitik dan Gerak Aktivitas Laut
    </p>
</div>
<p style="color: #64748b; font-size: 12px; font-weight: 700; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0.5px;">Pilih Halaman</p>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["Dashboard & Analisis", "Histori Prediksi", "Tentang Sistem"],
    label_visibility="collapsed"
)

if model is None:
    st.error("Gagal memuat model. Pastikan file 'random_forest_model_terbaik.joblib' tersedia di direktori aplikasi.")
else:
    if menu == "Dashboard & Analisis":
        col_form, col_res = st.columns([1.1, 0.9])
        
        with col_form:
            st.markdown("### 📥 Input Parameter Kapal")
            
            if st.session_state.reset_trigger:
                st.session_state.reset_trigger = False
            
            with st.form(key="input_kapal_form"):
                c1, c2 = st.columns(2)
                with c1:
                    fishing_hours = st.number_input(
                        "Fishing Hours (Decimal)",
                        min_value=0.0, max_value=24.0, value=9.7923, step=0.0001, format="%.4f"
                    )
                    fishing_efficiency_ratio = st.number_input(
                        "Fishing Efficiency Ratio",
                        min_value=0.0, max_value=10.0, value=0.00299, step=0.00001, format="%.5f"
                    )
                    day_of_week = st.slider(
                        "Day of Week (0=Senin, 6=Minggu)",
                        min_value=0, max_value=6, value=3
                    )
                    entry_session = st.slider(
                        "Entry Session",
                        min_value=0, max_value=5, value=1
                    )
                with c2:
                    total_trip_duration = st.number_input(
                        "Total Trip Duration (Hours)",
                        min_value=0.0, value=7.701, step=0.001, format="%.3f"
                    )
                    month = st.slider(
                        "Month",
                        min_value=1, max_value=12, value=3
                    )
                    is_weekend = st.selectbox(
                        "Is Weekend?",
                        options=[0, 1],
                        format_func=lambda x: "Weekday (0)" if x == 0 else "Weekend (1)"
                    )
                
                st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                btn_pred, btn_res = st.columns(2)
                with btn_pred:
                    submit_clicked = st.form_submit_button("🔍 Prediksi")
                with btn_res:
                    reset_clicked = st.form_submit_button("🔄 Reset Input")
                    if reset_clicked:
                        st.session_state.reset_trigger = True
                        st.rerun()

            input_data = pd.DataFrame(
                [[fishing_hours, total_trip_duration, fishing_efficiency_ratio, month, day_of_week, is_weekend, entry_session]],
                columns=MODEL_COLUMNS
            )
            
            with st.expander("👁️ Lihat Data Input Format Tabel", expanded=False):
                st.dataframe(input_data, use_container_width=True)

        with col_res:
            st.markdown("### 🎯 Hasil Klasifikasi")
            
            if submit_clicked:
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0] if hasattr(model, "predict_proba") else None
                details = get_cluster_details(prediction)
                
                new_record = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Fishing Hours": fishing_hours,
                    "Trip Duration": total_trip_duration,
                    "Efficiency Ratio": fishing_efficiency_ratio,
                    "Month": month,
                    "Day of Week": day_of_week,
                    "Is Weekend": is_weekend,
                    "Entry Session": entry_session,
                    "Cluster": f"Cluster {prediction}",
                    "Kategori": details["name"],
                    "Tingkat Risiko": details["risk"]
                }
                st.session_state.history.append(new_record)
                
                st.markdown(f"""
                <div class="result-card {details['class_css']}">
                    <h4 style='margin: 0 0 5px 0; color: #1e293b; font-size: 14px; font-weight:600; text-transform: uppercase;'>Nomor Cluster:</h4>
                    <h3 style='margin: 0 0 15px 0; color: #334155; font-size: 20px; font-weight: 700;'>Cluster {prediction}</h3>
                    <h4 style='margin: 0 0 5px 0; color: #1e293b; font-size: 14px; font-weight:600; text-transform: uppercase;'>Nama Kategori:</h4>
                    <h2 style='margin: 0 0 15px 0; color: {details['color']}; font-size: 24px; font-weight: 700;'>{details['name']}</h2>
                    <h4 style='margin: 0 0 5px 0; color: #1e293b; font-size: 14px; font-weight:600; text-transform: uppercase;'>Tingkat Risiko:</h4>
                    <p style='margin: 0 0 15px 0; font-size: 16px; font-weight: 600; color: {details['color']};'>{details['risk']}</p>
                    <hr style='margin: 15px 0; border: 0; border-top: 1px solid rgba(0,0,0,0.1);'>
                    <h4 style='margin: 0 0 5px 0; color: #1e293b; font-size: 14px; font-weight:600; text-transform: uppercase;'>Deskripsi Singkat:</h4>
                    <p style='margin: 0; font-size: 14px; line-height: 1.6; color: #334155;'>{details['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if proba is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("#### 📊 Nilai Probabilitas Klaster")
                    
                    labels_mapping = [get_cluster_details(c)["name"] for c in model.classes_]
                    proba_df = pd.DataFrame({
                        "Klaster": [f"Cluster {c}" for c in model.classes_],
                        "Kategori Risiko": labels_mapping,
                        "Probabilitas": [f"{p*100:.2f}%" for p in proba]
                    })
                    st.table(proba_df)
            else:
                st.info("Silakan masukkan parameter kapal di panel sebelah kiri lalu klik tombol prediksi untuk memproses data.")

        st.markdown("---")
        st.markdown("### 📈 Visualisasi Dashboard Analisis Historis")
        
        if len(st.session_state.history) > 0:
            df_hist = pd.DataFrame(st.session_state.history)
            
            v1, v2 = st.columns(2)
            with v1:
                fig_pie = px.pie(
                    df_hist, 
                    names='Kategori',
                    title='Proporsi Kategori Hasil Klasifikasi Sesi Ini',
                    color='Kategori',
                    color_discrete_map={
                        get_cluster_details(0)["name"]: get_cluster_details(0)["color"],
                        get_cluster_details(1)["name"]: get_cluster_details(1)["color"],
                        get_cluster_details(2)["name"]: get_cluster_details(2)["color"]
                    },
                    hole=0.4
                )
                fig_pie.update_layout(margin=dict(t=40, b=20, l=20, r=20), legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with v2:
                all_clusters = ["Cluster 0", "Cluster 1", "Cluster 2"]
                counts = df_hist['Cluster'].value_counts().reindex(all_clusters, fill_value=0)
                
                fig_bar = px.bar(
                    x=all_clusters,
                    y=counts.values,
                    labels={'x': 'Klaster Model', 'y': 'Frekuensi Kemunculan'},
                    title='Frekuensi Pendeteksian per Nomor Klaster',
                    color=all_clusters,
                    color_discrete_map={
                        "Cluster 0": get_cluster_details(0)["color"],
                        "Cluster 1": get_cluster_details(1)["color"],
                        "Cluster 2": get_cluster_details(2)["color"]
                    }
                )
                fig_bar.update_layout(showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig_bar, use_container_width=True)
                
            st.markdown("#### 📋 Ringkasan Distribusi Tingkat Risiko Saat Ini")
            r_high = len(df_hist[df_hist['Cluster'] == 'Cluster 0'])
            r_med  = len(df_hist[df_hist['Cluster'] == 'Cluster 1'])
            r_low  = len(df_hist[df_hist['Cluster'] == 'Cluster 2'])
            
            k1, k2, k3 = st.columns(3)
            k1.metric("🔴 TOTAL RISIKO TINGGI", f"{r_high} Analisis")
            k2.metric("🟡 TOTAL RISIKO SEDANG", f"{r_med} Analisis")
            k3.metric("🟢 TOTAL RISIKO RENDAH", f"{r_low} Analisis")
        else:
            st.info("Visualisasi grafik akan muncul secara otomatis setelah Anda melakukan simulasi prediksi data kapal beberapa kali.")

    elif menu == "Histori Prediksi":
        st.markdown("### 🗄️ Histori Rekam Jejak Prediksi")
        
        if len(st.session_state.history) > 0:
            df_hist = pd.DataFrame(st.session_state.history)
            
            h1, h2, h3 = st.columns([1, 1, 2])
            h1.metric("Total Prediksi Dilakukan", f"{len(df_hist)} Kali")
            
            with h2:
                st.markdown("<div style='padding-top:25px;'></div>", unsafe_allow_html=True)
                csv = df_hist.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh File Histori (CSV)",
                    data=csv,
                    file_name=f"histori_prediksi_siagalaut_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
            with h3:
                st.markdown("<div style='padding-top:25px;'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Hapus Bersih Semua Histori"):
                    st.session_state.history = []
                    st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.info("Belum ada data transaksi analisis yang terekam pada sesi running ini.")

    elif menu == "Tentang Sistem":
        st.markdown("### 📘 Tentang Sistem Pendeteksian")
        st.write(
            "Sistem ini dirancang khusus menggunakan algoritma klasifikasi ensemble "
            "untuk memetakan pergerakan spasial dan temporal kapal perikanan di wilayah perairan "
            "Provinsi Sulawesi Tenggara berdasarkan data transmisi AIS (Automatic Identification System)."
        )
        st.write(
            "Melalui rekayasa parameter durasi, efisiensi operasi, dan waktu entri wilayah, model dapat "
            "mengelompokkan tingkat kerawanan anomali aktivitas guna mengoptimalkan patroli instansi pengawas kemaritiman."
        )

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #94a3b8; font-size: 13px;'>"
    "Aplikasi Demonstrasi Klasifikasi Pola Aktivitas Kapal Perikanan Berdasarkan Data AIS Perairan Sultra • Berbasis Streamlit Modern"
    "</p>", 
    unsafe_allow_html=True
)