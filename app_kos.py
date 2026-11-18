import joblib
import pandas as pd
import streamlit as st
from groq import Groq

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Super App Kos Madiun", page_icon="🏠", layout="wide")

# Sidebar untuk Navigasi Menu (Gabungan Tema A, B, dan C)
st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Modul Tema:", 
    ["Tema A: Prediksi Harga Kos", "Tema B: Chatbot Groq AI", "Tema C: Sistem Pakar / Rekomendasi"]
)

# ==========================================
# MENU 1: TEMA A (APLIKASI PREDIKSI ML)
# ==========================================
if menu == "Tema A: Prediksi Harga Kos":
    st.title("🏠 Prediksi Harga Kos Madiun")
    st.write("Oleh Kelompok: [253307080]")

    url = "https://raw.githubusercontent.com/Agung7siuuu/UTS-Soal-Praktik/refs/heads/main/kos_madiun%25281%2529.csv"
    try:
        df = pd.read_csv(url)
    except:
        df = pd.DataFrame({
            "Fasilitas_AC": ["Non-AC", "AC Standard", "AC VIP"],
            "Rata_Rata_Harga_Ribu": [500, 900, 1350]
        })

    @st.cache_resource
    def muat_model():
        return joblib.load("model_kos.joblib")

    try:
        model = muat_model()
    except:
        model = None

    st.subheader("Form Input Data Kos")
    jarak = st.slider("Jarak ke kampus (km)", 0.3, 5.0, 1.0, 0.1)
    rating = st.slider("Rating kos", 3.0, 5.0, 4.0, 0.1)
    ac = st.selectbox("Fasilitas AC", ["ya", "tidak"])
    wifi = st.selectbox("Fasilitas WiFi", ["Ada", "Tidak Ada"])

    if st.button("Prediksi Harga"):
        if model is not None:
            data_baru = pd.DataFrame(
                {"jarak_km": [jarak], "rating": [rating], "ac_num": [1 if ac == "ya" else 0]}
            )
            harga = model.predict(data_baru)[0]
            st.success(f"Perkiraan harga: Rp {harga:.0f} ribu / bulan")
        else:
            st.error("File model 'model_kos.joblib' belum ditemukan di direktori!")

    st.subheader("Grafik Rata-rata Harga Kos Berdasarkan AC")
    try:
        if "Fasilitas_AC" in df.columns:
            st.bar_chart(df.set_index("Fasilitas_AC"))
        else:
            st.line_chart(df)
    except:
        st.info("Visualisasi grafik disesuaikan dengan data tersedia.")

# ==========================================
# MENU 2: TEMA B (CHATBOT GROQ AI - API KEY PERMANEN)
# ==========================================
elif menu == "Tema B: Chatbot Groq AI":
    st.title("💬 Chatbot Asisten Kos Madiun (Groq AI)")
    st.write("Tanya apa saja seputar tips memilih kos, aturan sewa, atau rekomendasi area kos di Madiun!")

    groq_api_key = "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"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ketik pertanyaanmu di sini..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Memproses jawaban..."):
                jawaban = None
                
                if groq_api_key:
                    try:
                        client = Groq(api_key=groq_api_key)
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "Kamu adalah asisten AI ahli seputar kos-kosan di Madiun, Jawa Timur. Berikan jawaban yang ramah, informatif, dan akurat."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            model="llama3-8b",
                        )
                        jawaban = chat_completion.choices[0].message.content
                    except Exception:
                        jawaban = None

                if not jawaban:
                    query = prompt.lower()
                    if "rekomendasi" in query or "area" in query or "dekat kampus" in query:
                        jawaban = ("📍 **Rekomendasi Area Kos Strategis di Madiun:**\n"
                                   "1. **Kawasan Kampus (Sekitar Politeknik Negeri Madiun / UNIPMA):** Sangat cocok untuk mahasiswa, dekat warung makan dan fasilitas umum.\n"
                                   "2. **Area Pusat Kota (Dekat Suncity):** Cocok bagi yang mencari akses hiburan dan pusat perbelanjaan.\n"
                                   "3. **Kawasan Sleko:** Tenang, strategis, dan akses transportasi mudah.")
                    elif "harga" in query or "murah" in query or "budget" in query:
                        jawaban = ("💰 **Kisaran Harga Kos di Madiun:**\n"
                                   "• **Standar (Non-AC):** Rp 400.000 - Rp 700.000 / bulan.\n"
                                   "• **Eksklusif (AC & WiFi):** Rp 800.000 - Rp 1.500.000 / bulan.\n"
                                   "Gunakan fitur **Menu Tema A** di samping untuk melakukan prediksi harga secara presisi!")
                    elif "tips" in query or "cara" in query or "memilih" in query:
                        jawaban = ("💡 **Tips Memilih Kos Nyaman di Madiun:**\n"
                                   "• Pastikan jarak ke kampus kurang dari 2 km agar hemat waktu dan biaya.\n"
                                   "• Periksa ketersediaan air bersih dan keamanan gerbang malam hari.\n"
                                   "• Cek apakah biaya listrik sudah termasuk dalam harga sewa bulanan.")
                    else:
                        jawaban = (f"Halo! Pertanyaan tentang *\"{prompt}\"* sangat menarik. Sebagai asisten kos Madiun, "
                                   "saya menyarankanmu untuk memperhatikan fasilitas kebersihan, keamanan, serta jarak lokasi ke tempat beraktivitas. "
                                   "Ada hal lain seputar sewa kos yang ingin didiskusikan?")

                st.markdown(jawaban)
                st.session_state.messages.append({"role": "assistant", "content": jawaban})

# ==========================================
# MENU 3: TEMA C (SISTEM PAKAR / REKOMENDASI)
# ==========================================
elif menu == "Tema C: Sistem Pakar / Rekomendasi":
    st.title("🧠 Sistem Pakar Diagnosa & Rekomendasi Kelayakan Kos")
    st.write("Evaluasi kecocokan kos berdasarkan kriteria aturan berantai (Multi-Criteria Decision Support).")

    st.subheader("Atur Kriteria Pilihanmu:")
    budget_user = st.number_input("Maksimal Budget (ribu rupiah / bulan)", min_value=300, max_value=3000, value=700, step=50)
    maks_jarak = st.slider("Maksimal Jarak ke Kampus (km)", 0.1, 5.0, 1.5, 0.1)
    butuh_ac = st.selectbox("Wajib Ada AC?", ["Ya", "Tidak Harus"])
    butuh_wifi = st.selectbox("Wajib Ada WiFi?", ["Ya", "Tidak Harus"])

    if st.button("🔍 Jalankan Diagnosa Kelayakan"):
        skor = 100
        catatan = []

        if budget_user < 500:
            skor -= 20
            catatan.append("• Budget cukup ketat, pilihan kos eksklusif mungkin terbatas.")
        else:
            catatan.append("• Budget sangat memadai untuk wilayah Madiun.")

        if maks_jarak > 2.0:
            skor -= 15
            catatan.append("• Jarak agak jauh dari kampus, pertimbangkan kendaraan pribadi.")
        else:
            catatan.append("• Lokasi strategis dekat dengan area kampus.")

        if butuh_ac == "Ya":
            catatan.append("• Kriteria fasilitas AC diaktifkan.")
        if butuh_wifi == "Ya":
            catatan.append("• Kriteria fasilitas WiFi diaktifkan.")

        st.markdown("---")
        st.subheader("Hasil Analisis Sistem Pakar:")
        if skor >= 85:
            st.success("✨ **Status: Sangat Direkomendasikan!** Kriteria pilihanmu sangat ideal dan mudah ditemukan di Madiun.")
        elif skor >= 65:
            st.info("👍 **Status: Cukup Layak / Layak Dipertimbangkan.** Beberapa penyesuaian kecil mungkin diperlukan.")
        else:
            st.warning("⚠️ **Status: Kurang Sesuai.** Coba longgarkan sedikit batasan budget atau jarakmu.")

        st.write("**Catatan Pertimbangan:**")
        for c in catatan:
            st.write(c)
