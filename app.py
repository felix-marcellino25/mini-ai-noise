# app.py
import streamlit as st
import joblib
import numpy as np
import librosa
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import os
import tempfile

# --- Konfigurasi InfluxDB (ganti dengan milikmu) ---
INFLUXDB_URL = "https://us-west-2-1.aws.cloud2.influxdata.com"
INFLUXDB_TOKEN = st.secrets["INFLUXDB_TOKEN"] if "INFLUXDB_TOKEN" in st.secrets else "2MuIXyIyj8YHMLEH6bLjUxcRyWie8RowDMmhgSLT580j7y4KFCHdIajbw0rdav1OaMlBP3YqzNMaFM9LKPWB9Q=="
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
st.title("🔊 Mini AI: Klasifikasi Kebisingan")
st.write("Pilih cara input: masukkan nilai dB atau upload file audio (.wav).")

tab1, tab2 = st.tabs(["Input Nilai dB", "Upload File Audio"])

# Tab 1: Input Manual
with tab1:
    noise_dB = st.number_input("Level Kebisingan (dB)", min_value=0.0, max_value=140.0, value=85.0, step=1.0)
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
st.caption("📌 Estimasi dB dari file audio bersifat **relatif** dan **tidak terkalibrasi**. Untuk hasil akurat, gunakan SPL meter terkalibrasi.")
