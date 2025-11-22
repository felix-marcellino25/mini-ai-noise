# app.py
import streamlit as st
import joblib
import numpy as np
import librosa
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import os
import tempfile
import base64

# --- Informasi Mahasiswa ---
NAMA = "Felix Marcellino Henrikus"
NIM = "632025006"
PRODI = "Magister Sains Data"
FAKULTAS = "Fakultas Sains dan Matematika"
UNIVERSITAS = "Universitas Kristen Satya Wacana Salatiga"

# --- Tampilan Header ---
st.markdown(f"""
<div style="display: flex; align-items: center; gap: 15px; padding: 10px; background-color: #f9f9fb; border-bottom: 1px solid #ddd;">
    <div style="flex-shrink: 0;">
        <img src="data:image/png;base64,{base64.b64encode(open('logo_univ.png', 'rb').read()).decode()}" width="100" />
    </div>
    <div style="flex-shrink: 0;">
        <img src="data:image/png;base64,{base64.b64encode(open('logo_fakultas.png', 'rb').read()).decode()}" width="100" />
    </div>
    <div style="flex-grow: 1; text-align: left; margin-left: 10px;">
        <h2 style="margin: 0; color: #435da3;">{NAMA} • {NIM}</h2>
        <p style="margin: 4px 0; color: #435da3;"><strong>{PRODI} • {FAKULTAS}</strong></p>
        <p style="margin: 0; color: #435da3;">{UNIVERSITAS}</p>       
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
<style>
    .main {
        background-color: #cbcbd4;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# --- Konfigurasi InfluxDB (ganti dengan milikmu) ---
INFLUXDB_URL = "https://us-east-1-1.aws.cloud2.influxdata.com"
INFLUXDB_TOKEN = st.secrets["INFLUXDB_TOKEN"] if "INFLUXDB_TOKEN" in st.secrets else "-9LaPT1-_kT-Pp7k-StlXSzkuUPBWXskzeBO7cI1J1jBQ-txIZueGCz0igXqejlcYqA53V0l4QMBi-AQAGA34Q=="
INFLUXDB_ORG = "3f99312a36600ebf"
INFLUXDB_BUCKET = "UjiCobaAI2025"

# Untuk development lokal, bisa hardcode token (jangan commit ke GitHub!)
# INFLUXDB_TOKEN = "x8aBc...xyz"

# --- Load model ---
@st.cache_resource
def load_model():
    return joblib.load("noise_classifier_best.pkl")

model = load_model()

# --- Lookup durasi ---
NOISE_DATA = {
    80: 480, 85: 480, 88: 240, 91: 120, 94: 60, 97: 30,
    100: 15, 103: 7.5, 106: 3.75, 109: 1.88, 112: 0.94,
    115: 0.46, 118: 0.23, 121: 0.11, 124: 0.05
}

def get_closest_duration(dB):
    return min(NOISE_DATA.keys(), key=lambda x: abs(x - dB))

# --- Kirim ke InfluxDB ---
def save_to_influxdb(noise_dB, is_safe, source="manual"):
    try:
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        
        point = Point("noise_classification") \
            .tag("source", source) \
            .field("noise_level_dB", float(noise_dB)) \
            .field("is_safe", bool(is_safe)) \
            .field("max_safe_minutes", float(NOISE_DATA.get(get_closest_duration(noise_dB), 0)) if not is_safe else 0.0)
        
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
        client.close()
        return True
    except Exception as e:
        st.warning(f"⚠️ Gagal simpan ke InfluxDB: {e}")
        return False

# --- Ekstraksi dB dari audio (estimasi kasar) ---
def estimate_db_from_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    rms = np.sqrt(np.mean(y**2))
    # Asumsi: 0 dBFS ≈ 94 dB SPL (kalibrasi kasar)
    # Jika audio sudah dinormalisasi, ini hanya RELATIF
    if rms == 0:
        return 0
    db = 20 * np.log10(rms) + 94
    return max(0, db)

# --- UI ---
st.markdown("<h1 style='text-align: center; color: #435da3;'>🔊 Arsitektur Artificial Intelligence untuk Klasifikasi Kebisingan menggunakan Machine Learning</h1>", unsafe_allow_html=True)
st.write("Pilih cara input: masukkan nilai kebisingan atau upload file audio (dalam format .wav).")

tab1, tab2 = st.tabs(["Input Nilai dB", "Upload File Audio"])

# Tab 1: Input Manual
with tab1:
    noise_dB = st.number_input("Level Kebisingan (dB)", min_value=0.0, max_value=150.0, value=80.0, step=0.5)
    if st.button("Klasifikasi (dB)"):
        is_safe = model.predict([[noise_dB]])[0]
        status = "✅ Aman" if is_safe else "⚠️ Tidak Aman"
        st.subheader(f"Hasil: {status}")
        if not is_safe:
            closest_key = get_closest_duration(noise_dB)
            max_minutes = NOISE_DATA[closest_key]
            st.write(f"⏱️ Durasi maksimal aman: **{max_minutes:.2f} menit**")
        if save_to_influxdb(noise_dB, is_safe, "manual"):
            st.success("✅ Data berhasil disimpan ke database.")

# Tab 2: Upload Audio
with tab2:
    uploaded_file = st.file_uploader("Upload file audio (.wav)", type=["wav"])
    if uploaded_file is not None:
        # Simpan ke file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        try:
            estimated_dB = estimate_db_from_audio(tmp_path)
            st.write(f"🎧 Estimasi level kebisingan: **{estimated_dB:.1f} dB**")
            
            is_safe = model.predict([[estimated_dB]])[0]
            status = "✅ Aman" if is_safe else "⚠️ Tidak Aman"
            st.subheader(f"Hasil: {status}")
            if not is_safe:
                closest_key = get_closest_duration(estimated_dB)
                max_minutes = NOISE_DATA[closest_key]
                st.write(f"⏱️ Durasi maksimal aman: **{max_minutes:.2f} menit**")
            if save_to_influxdb(estimated_dB, is_safe, "audio"):
                st.success("✅ Data berhasil disimpan ke database.")
        except Exception as e:
            st.error(f"❌ Gagal memproses audio: {e}")
        finally:
            os.unlink(tmp_path)

# --- Catatan kecil ---
st.markdown("---")
st.markdown("""
### 📌 Tentang Proyek Ini

Proyek ini merupakan tugas mata kuliah **Artificial Intelligence** yang bertujuan untuk membangun sistem AI yang mampu mengklasifikasikan tingkat kebisingan sebagai “aman” atau “tidak aman”, serta memberikan rekomendasi durasi maksimal paparan kebisingan yang aman bagi pendengaran manusia.
Sistem ini dibangun menggunakan:
- **Machine Learning** (model *Decision Tree*) untuk klasifikasi
- **Streamlit** sebagai dashboard interaktif
- **InfluxDB Cloud** sebagai *database* penyimpanan histori

### 👨‍🏫 Dosen Pengampu:
- Dr. Suryasatriya Trihandaru, M.Sc.nat.
- Prof. Hanna Arini Parhusip, M.Sc.nat.
- Dr. Bambang Susanto, MS.
- Prof. Eko Sediyono, M.Kom.
- Denny Indrajaya, M.Si.D.

### 📌 Rekomendasi
Estimasi kebisingan dari file audio belum terkalibrasi, gunakan SPL meter yang terkalibrasi.
""", unsafe_allow_html=True)
