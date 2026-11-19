# 🏠 MadiunKos — Aplikasi Info, Prediksi Harga & Sistem Cerdas Kos Madiun

Mini Proyek Mata Kuliah Praktikum — **Politeknik Negeri Madiun**  
**Nama:** Satria Agung Nur Ikhsan  
**NIM / Kelas:** 253307080 / Kelas 3C (D3 Teknologi Informasi)

## 🔗 Tautan Aplikasi (Live Demo)
Akses aplikasi web interaktif di Streamlit Community Cloud:  
👉 [Buka MadiunKos di Streamlit Cloud](https://prediksi-harga-kos-madiun-z4nfe5ee8npsayzudc2hn5.streamlit.app)

## ✨ Fitur Utama Aplikasi
Aplikasi dibangun menggunakan **Streamlit** dengan navigasi menu samping (*sidebar*) dan nuansa hijau segar modern:
1. **Katalog Kos:** Menampilkan daftar unit kos di Madiun dari `kos_madiun.csv` dalam tata letak kartu 2 kolom, lengkap dengan pencarian nama, filter fasilitas **AC** dan **WiFi**, jarak ke kampus, rating, serta tarif bulanan.
2. **Prediksi Harga:** Memperkirakan tarif sewa bulanan berdasarkan jarak ke kampus, rating, dan fasilitas AC/WiFi menggunakan `model_kos.joblib` (dengan *fallback* estimasi otomatis jika model belum dimuat) serta visualisasi grafik perbandingan tarif.
3. **Asisten FAQ & RAG:** Asisten tanya-jawab berbasis **Groq LLM** yang menggabungkan basis pengetahuan FAQ kos Madiun dan ringkasan data dari `kos_madiun.csv`.
4. **Rekomendasi Pakar:** Sistem pakar berbasis penalaran aturan berantai (*forward chaining*) untuk mencocokkan anggaran, kebutuhan fasilitas, dan lokasi kos.

## 📁 Struktur Berkas Proyek
- `app_kos.py` — Kode utama aplikasi Streamlit.
- `kos_madiun.csv` — Dataset data kos di wilayah Madiun.
- `model_kos.joblib` — Model *machine learning* untuk prediksi harga kos.
- `requirements.txt` — Daftar dependensi Python untuk *deployment*.
- `README.md` — Dokumentasi proyek.

## 💻 Menjalankan Secara Lokal
```bash
pip install -r requirements.txt
streamlit run app_kos.py
