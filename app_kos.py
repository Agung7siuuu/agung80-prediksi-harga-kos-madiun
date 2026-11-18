import joblib
import pandas as pd
import streamlit as st

# Judul aplikasi & identitas
st.title("🏠 Prediksi Harga Kos Madiun")
st.write("Oleh Kelompok: [253307080]")

# Memuat dataset untuk grafik langsung dari GitHub
url = "https://raw.githubusercontent.com/Agung7siuuu/UTS-Soal-Praktik/refs/heads/main/kos_madiun%20%281%29.csv"
df = pd.read_csv(url)


# Dekorator cache untuk memuat model joblib
@st.cache_resource
def muat_model():
  return joblib.load("model_kos.joblib")


model = muat_model()

# Bagian Input Form
st.subheader("Form Input Data Kos")
jarak = st.slider("Jarak ke kampus (km)", 0.3, 5.0, 1.0, 0.1)
rating = st.slider("Rating kos", 3.0, 5.0, 4.0, 0.1)
ac = st.selectbox("Fasilitas AC", ["ya", "tidak"])

# Input tambahan (Kegiatan 4)
wifi = st.selectbox("Fasilitas WiFi", ["Ada", "Tidak Ada"])

# Tombol Prediksi
if st.button("Prediksi Harga"):
  data_baru = pd.DataFrame(
      {"jarak_km": [jarak], "rating": [rating], "ac_num": [1 if ac == "ya" else 0]}
  )

  harga = model.predict(data_baru)[0]
  st.success(f"Perkiraan harga: Rp {harga:.0f} ribu / bulan")

# Grafik (Kegiatan 4)
st.subheader("Grafik Rata-rata Harga Kos Berdasarkan AC")
rata_ac = df.groupby("ac")["harga"].mean()
st.bar_chart(rata_ac)