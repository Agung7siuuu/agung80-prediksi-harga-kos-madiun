# 🏠 Super App Kos Madiun (Gabungan Tema A, B, & C)

Mini Proyek Mata Kuliah Praktikum - Politeknik Negeri Madiun  
**NIM / Kelas:** 253307080 / Kelas 3C (D3 Teknologi Informasi)

---

## 🔗 Tautan Aplikasi (Live Demo)
Akses aplikasi web interaktif yang telah di-deploy secara online:
👉 **[Link Streamlit Cloud](https://prediksi-harga-kos-madiun-edmpuubsvgppntpkzoflt4.streamlit.app)**

---

## 🚀 Fitur Utama Aplikasi
Aplikasi ini digabungkan menjadi satu platform web interaktif menggunakan framework **Streamlit**:
1. **Tema A (Prediksi Harga Kos):** Menggunakan model Machine Learning (`model_joblib`) untuk memprediksi estimasi harga sewa kos di wilayah Madiun berdasarkan jarak ke kampus, rating, fasilitas AC, dan WiFi, lengkap dengan visualisasi grafik data.
2. **Tema B (Chatbot Groq AI):** Asisten virtual berbasis LLM Groq (`llama3-8b`) dengan integrasi *fallback* cerdas untuk membantu menjawab pertanyaan seputar tips memilih kos, aturan sewa, hingga rekomendasi area strategis di Madiun.
3. **Tema C (Sistem Pakar / Rekomendasi):** Sistem pendukung keputusan berbasis kriteria (*Multi-Criteria Decision Support*) untuk mengevaluasi kelayakan dan kecocokan kos berdasarkan budget dan kebutuhan pengguna.

---

## 📁 Struktur Berkas di Repository
* `app_kos.py` : Berkas utama antarmuka web interaktif (menggabungkan Tema A, B, dan C).
* `model_kos.joblib` : Berkas model Machine Learning regresi yang telah dilatih.
* `kos_madiun.csv` : Dataset referensi harga kos di wilayah Madiun.
* `requirements.txt` : Daftar pustaka/dependensi pustaka Python yang dibutuhkan (`streamlit`, `pandas`, `joblib`, `scikit-learn`, `groq`).

---

## 💻 Cara Menjalankan Secara Lokal
1. Clone repository ini ke perangkat komputer Anda:
   ```bash
   git clone [https://github.com/Agung7siuuu/prediksi-harga-kos-madiun.git](https://github.com/Agung7siuuu/prediksi-harga-kos-madiun.git)
