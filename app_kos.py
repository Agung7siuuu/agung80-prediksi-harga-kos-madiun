import streamlit as st  # Mengimpor library Streamlit untuk membuat antarmuka aplikasi web
import pandas as pd  # Mengimpor pandas untuk mengolah data tabel / CSV dan DataFrame
import joblib  # Mengimpor joblib untuk memuat model machine learning yang sudah disimpan (.joblib)
import os  # Mengimpor os untuk mengecek keberadaan file di folder proyek
from groq import Groq  # Mengimpor klien Groq untuk memanggil model LLM (AI chatbot/FAQ)

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(  # Mengatur konfigurasi dasar halaman web Streamlit
    page_title="MadiunKos — Cari & Prediksi Hunian Madiun",  # Judul tab di browser
    page_icon="🏠",  # Ikon tab di browser
    layout="wide",  # Menggunakan lebar layar penuh (wide layout)
    initial_sidebar_state="expanded"  # Membuka sidebar secara default saat aplikasi pertama dibuka
)

BRAND_SVG = (  # Menyimpan kode SVG logo rumah hijau ala Bibit dalam bentuk string
    '<svg width="38" height="38" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">'  # Elemen pembungkus SVG ukuran 38x38 piksel
    '<rect width="38" height="38" rx="11" fill="#00B87C"/>'  # Kotak latar belakang berwarna hijau segar (#00B87C) dengan sudut membulat
    '<path d="M11 19.2L19 12L27 19.2V27H21.2V21.5H16.8V27H11V19.2Z" fill="#FFFFFF"/>'  # Bentuk siluet rumah berwarna putih
    '<circle cx="28.5" cy="10.5" r="3.5" fill="#B8F3DF"/>'  # Lingkaran aksen hijau muda di pojok kanan atas logo
    '</svg>'  # Menutup tag SVG
)

st.markdown("""  # Menyisipkan kode CSS kustom ke halaman Streamlit agar tampilan menyerupai aplikasi modern
<style>  # Awal blok gaya CSS
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');  # Memuat font Plus Jakarta Sans dari Google Fonts

    html, body, [class*="css"] {  # Menerapkan font ke seluruh elemen halaman
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;  # Keluarga font utama dengan cadangan sistem
    }

    header[data-testid="stHeader"] {  # Mengatur header bawaan Streamlit
        background: transparent !important;  # Membuat latar header transparan
    }
    [data-testid="stDecoration"] {  # Menargetkan garis dekorasi atas Streamlit
        display: none;  # Menyembunyikan garis dekorasi bawaan
    }

    .stApp {  # Latar belakang utama aplikasi
        background-color: #f7fbf9;  # Warna hijau keputihan yang lembut ala Bibit
        color: #111827;  # Warna teks utama gelap agar kontras
    }

    /* Sidebar putih bersih ala Bibit */
    section[data-testid="stSidebar"] {  # Panel sidebar di sisi kiri
        background-color: #ffffff !important;  # Latar putih bersih
        border-right: 1px solid #e5efea !important;  # Garis pembatas tipis di kanan sidebar
    }
    section[data-testid="stSidebar"] .block-container {  # Jarak dalam konten sidebar
        padding-top: 1.35rem;  # Padding atas
        padding-bottom: 2rem;  # Padding bawah
    }
    .sb-brand {  # Kotak identitas merek di bagian atas sidebar
        display: flex;  # Susunan fleksibel horizontal
        align-items: center;  # Rata tengah secara vertikal
        gap: 12px;  # Jarak antar elemen
        padding-bottom: 15px;  # Jarak bawah
        margin-bottom: 15px;  # Margin bawah
        border-bottom: 1px solid #eef5f1;  # Garis pemisah tipis
    }
    .sb-name {  # Judul merek MadiunKos
        font-size: 1.15rem;  # Ukuran huruf
        font-weight: 800;  # Ketebalan huruf ekstra tebal
        color: #111827;  # Warna teks gelap
        letter-spacing: -0.02em;  # Perapatan jarak antar huruf sedikit
        line-height: 1.1;  # Tinggi baris rapat
    }
    .sb-tag {  # Subjudul kecil di bawah nama merek
        font-size: 0.73rem;  # Ukuran huruf kecil
        font-weight: 700;  # Huruf tebal
        color: #00b87c;  # Warna hijau utama Bibit
        margin-top: 2px;  # Jarak atas kecil
    }
    .sb-sec {  # Label judul bagian menu di sidebar
        font-size: 0.7rem;  # Ukuran huruf kecil
        font-weight: 700;  # Huruf tebal
        text-transform: uppercase;  # Huruf kapital semua
        letter-spacing: 0.07em;  # Jarak antar huruf
        color: #6b7280;  # Warna abu-abu sedang
        margin-bottom: 8px;  # Jarak bawah
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] {  # Wadah pilihan radio navigasi di sidebar
        display: flex;  # Susunan flex
        flex-direction: column;  # Susun ke bawah secara vertikal
        gap: 6px;  # Jarak antar item menu
        background: transparent !important;  # Latar transparan
        padding: 0 !important;  # Tanpa padding luar
        border: none !important;  # Tanpa border luar
        box-shadow: none !important;  # Tanpa bayangan
        margin-bottom: 18px;  # Jarak bawah sebelum info status
        width: 100%;  # Lebar penuh sidebar
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {  # Setiap tombol menu navigasi
        background-color: #ffffff !important;  # Latar putih
        border: 1px solid #e5efea !important;  # Border tipis hijau pucat
        border-radius: 10px !important;  # Sudut membulat
        padding: 10px 13px !important;  # Ruang dalam tombol menu
        cursor: pointer;  # Kursor tangan saat diarahkan
        width: 100%;  # Lebar penuh
        transition: all 0.15s ease;  # Animasi transisi halus
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label * {  # Teks di dalam menu radio
        color: #374151 !important;  # Warna teks abu gelap agar jelas
        font-size: 0.88rem !important;  # Ukuran huruf menu
        font-weight: 600 !important;  # Ketebalan semi-bold
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {  # Saat kursor mengarah ke menu
        background-color: #e6f9f2 !important;  # Latar hijau mint lembut
        border-color: #9ce8ce !important;  # Border hijau lebih tegas
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover * {  # Warna teks saat hover
        color: #008a5c !important;  # Hijau tua kontras
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {  # Menu yang sedang aktif dipilih
        background-color: #e6f9f2 !important;  # Latar hijau mint aktif
        border-color: #00b87c !important;  # Border hijau utama
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] *,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked),
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) * {  # Teks menu yang sedang aktif
        color: #008a5c !important;  # Warna hijau gelap jelas terbaca
        font-weight: 800 !important;  # Huruf ekstra tebal
    }

    .sb-box {  # Kotak ringkasan info dataset & model di sidebar
        background: #f4fbf8;  # Latar hijau sangat muda
        border: 1px solid #d4f2e6;  # Border hijau lembut
        border-radius: 12px;  # Sudut membulat
        padding: 13px 14px;  # Ruang dalam kotak
        font-size: 0.78rem;  # Ukuran huruf kecil
        color: #374151;  # Warna teks
        line-height: 1.65;  # Jarak antar baris
    }
    .sb-box b {  # Teks tebal di dalam kotak info sidebar
        color: #008a5c;  # Warna hijau kontras
    }

    .block-container {  # Kontainer utama konten halaman
        padding-top: 1.3rem;  # Jarak atas
        padding-bottom: 3rem;  # Jarak bawah
        max-width: 1180px;  # Batas lebar maksimal konten
    }

    /* Hero banner hijau Bibit */
    .app-hero {  # Banner judul halaman di bagian atas
        background: linear-gradient(120deg, #00b87c 0%, #00a36e 60%, #10b981 100%);  # Gradasi hijau segar
        border-radius: 18px;  # Sudut membulat besar
        padding: 22px 26px;  # Ruang dalam banner
        color: #ffffff;  # Warna teks putih
        display: flex;  # Susunan fleksibel
        align-items: center;  # Rata tengah vertikal
        justify-content: space-between;  # Pisahkan kiri dan kanan
        flex-wrap: wrap;  # Bungkus ke baris baru jika layar sempit
        gap: 14px;  # Jarak antar elemen
        margin-bottom: 16px;  # Jarak bawah
        box-shadow: 0 10px 24px -12px rgba(0, 184, 124, 0.45);  # Bayangan hijau lembut
    }
    .app-hero-title {  # Judul utama pada banner
        font-size: 1.4rem;  # Ukuran huruf judul
        font-weight: 800;  # Huruf sangat tebal
        letter-spacing: -0.02em;  # Sedikit rapat antar huruf
        color: #ffffff;  # Teks putih
    }
    .app-hero-sub {  # Deskripsi singkat di bawah judul banner
        font-size: 0.85rem;  # Ukuran deskripsi
        color: #e6f9f2;  # Putih kehijauan lembut
        margin-top: 4px;  # Jarak atas kecil
    }
    .hero-pill {  # Badge lokasi di kanan banner
        background: rgba(255, 255, 255, 0.22);  # Latar putih semi-transparan
        border: 1px solid rgba(255, 255, 255, 0.38);  # Border putih tipis
        color: #ffffff;  # Teks putih
        font-size: 0.78rem;  # Ukuran teks badge
        font-weight: 700;  # Huruf tebal
        padding: 7px 14px;  # Ruang dalam badge
        border-radius: 999px;  # Bentuk kapsul/pill
    }

    /* Stat strip */
    .stat-tile {  # Kartu metrik/statistik ringkas (4 kolom)
        background: #ffffff;  # Latar putih
        border: 1px solid #e5efea;  # Border tipis
        border-radius: 14px;  # Sudut membulat
        padding: 14px 16px;  # Ruang dalam
        box-shadow: 0 2px 6px rgba(0, 184, 124, 0.04);  # Bayangan sangat halus
    }
    .stat-k {  # Label metrik (misal: Unit Tersedia)
        font-size: 0.72rem;  # Ukuran kecil
        font-weight: 700;  # Tebal
        text-transform: uppercase;  # Huruf kapital
        letter-spacing: 0.04em;  # Spasi huruf
        color: #6b7280;  # Abu-abu
    }
    .stat-v {  # Nilai utama metrik (misal: Rp 600.000)
        font-size: 1.25rem;  # Ukuran angka besar
        font-weight: 800;  # Sangat tebal
        color: #111827;  # Warna gelap
        margin-top: 4px;  # Jarak atas
    }

    /* Panel section */
    .sec-card {  # Kartu panel pembungkus fitur
        background: #ffffff;  # Latar putih
        border: 1px solid #e5efea;  # Border tipis
        border-radius: 16px;  # Sudut membulat
        padding: 20px 22px;  # Ruang dalam
        margin-bottom: 16px;  # Jarak bawah
        box-shadow: 0 2px 8px rgba(17, 24, 39, 0.03);  # Bayangan lembut
    }
    .sec-head {  # Judul di dalam kartu panel
        font-size: 1.02rem;  # Ukuran huruf
        font-weight: 800;  # Tebal
        color: #111827;  # Warna gelap
        margin-bottom: 2px;  # Jarak bawah kecil
    }
    .sec-sub {  # Subjudul/keterangan di dalam panel
        font-size: 0.82rem;  # Ukuran kecil
        color: #6b7280;  # Abu-abu
        margin-bottom: 12px;  # Jarak bawah
    }

    /* Kartu listing grid 2 kolom */
    .prop-card {  # Kartu unit kos di katalog
        background: #ffffff;  # Latar putih
        border: 1px solid #e5efea;  # Border tipis
        border-radius: 16px;  # Sudut membulat
        overflow: hidden;  # Memotong elemen anak agar sudut atas tetap membulat
        margin-bottom: 14px;  # Jarak bawah antar kartu
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;  # Efek transisi
    }
    .prop-card:hover {  # Efek kartu saat diarahkan kursor
        transform: translateY(-2px);  # Sedikit terangkat ke atas
        border-color: #00b87c;  # Border berubah hijau utama
        box-shadow: 0 10px 20px -12px rgba(0, 184, 124, 0.25);  # Bayangan hijau
    }
    .prop-banner {  # Garis aksen hijau di bagian paling atas kartu kos
        height: 6px;  # Tinggi garis
        background: linear-gradient(90deg, #00b87c 0%, #34d399 100%);  # Gradasi hijau
    }
    .prop-body {  # Isi konten kartu unit kos
        padding: 16px 18px;  # Ruang dalam
    }
    .prop-top {  # Baris atas kartu (nama kos & rating)
        display: flex;  # Susunan flex
        justify-content: space-between;  # Pisahkan kiri dan kanan
        align-items: flex-start;  # Rata atas
        gap: 10px;  # Jarak antar elemen
    }
    .prop-title {  # Nama kos
        font-size: 1.02rem;  # Ukuran nama kos
        font-weight: 800;  # Tebal
        color: #111827;  # Warna gelap
    }
    .prop-rating {  # Badge rating bintang
        font-size: 0.76rem;  # Ukuran kecil
        font-weight: 700;  # Tebal
        color: #008a5c;  # Hijau tua
        background: #e6f9f2;  # Hijau mint muda
        border: 1px solid #b8f3df;  # Border hijau muda
        padding: 3px 9px;  # Ruang dalam badge
        border-radius: 999px;  # Bentuk kapsul
        white-space: nowrap;  # Teks rating tidak terpotong ke baris baru
    }
    .prop-loc {  # Informasi jarak ke kampus
        font-size: 0.82rem;  # Ukuran teks
        color: #6b7280;  # Abu-abu
        margin-top: 4px;  # Jarak atas
    }
    .prop-badges {  # Wadah badge fasilitas AC / WiFi
        margin-top: 10px;  # Jarak atas
        margin-bottom: 14px;  # Jarak bawah
    }
    .p-badge {  # Gaya dasar badge fasilitas
        display: inline-block;  # Tampil sejajar horizontal
        font-size: 0.72rem;  # Ukuran kecil
        font-weight: 700;  # Tebal
        color: #4b5563;  # Abu gelap
        background: #f3f4f6;  # Abu muda
        border: 1px solid #e5e7eb;  # Border abu
        padding: 3px 10px;  # Ruang dalam
        border-radius: 999px;  # Bentuk kapsul
        margin-right: 6px;  # Jarak kanan antar badge
    }
    .p-badge.yes {  # Gaya badge jika fasilitas tersedia
        color: #008a5c;  # Hijau tua
        background: #e6f9f2;  # Hijau mint
        border-color: #9ce8ce;  # Border hijau
    }
    .prop-foot {  # Bagian bawah kartu kos (harga & tipe kamar)
        display: flex;  # Susunan flex
        justify-content: space-between;  # Pisahkan kiri dan kanan
        align-items: center;  # Rata tengah vertikal
        padding-top: 12px;  # Jarak atas
        border-top: 1px solid #f0f5f2;  # Garis pemisah tipis di atas footer kartu
    }
    .prop-price {  # Nominal harga sewa bulanan
        font-size: 1.08rem;  # Ukuran harga
        font-weight: 800;  # Sangat tebal
        color: #00b87c;  # Hijau utama Bibit
    }
    .prop-price span {  # Label "/ bulan" di sebelah harga
        font-size: 0.76rem;  # Ukuran lebih kecil
        font-weight: 500;  # Normal
        color: #6b7280;  # Abu-abu
    }
    .prop-tag-city {  # Label tipe kamar di kanan bawah kartu
        font-size: 0.72rem;  # Ukuran kecil
        font-weight: 700;  # Tebal
        color: #008a5c;  # Hijau tua
        background: #f4fbf8;  # Hijau sangat muda
        border: 1px solid #d4f2e6;  # Border hijau muda
        padding: 3px 9px;  # Ruang dalam
        border-radius: 999px;  # Bentuk kapsul
    }

    /* Box estimasi / hasil hijau Bibit */
    .est-highlight {  # Kotak sorotan hasil prediksi harga atau rekomendasi pakar
        background: linear-gradient(135deg, #f4fbf8 0%, #e6f9f2 100%);  # Gradasi hijau lembut
        border: 1.5px solid #00b87c;  # Border hijau utama
        border-radius: 16px;  # Sudut membulat
        padding: 22px;  # Ruang dalam
    }
    .est-lbl {  # Label kecil di atas hasil prediksi/rekomendasi
        font-size: 0.74rem;  # Ukuran kecil
        font-weight: 800;  # Sangat tebal
        text-transform: uppercase;  # Huruf kapital
        letter-spacing: 0.05em;  # Spasi huruf
        color: #008a5c;  # Hijau tua
    }
    .est-val {  # Angka besar hasil estimasi harga
        font-size: 2.05rem;  # Ukuran angka besar
        font-weight: 800;  # Sangat tebal
        color: #111827;  # Warna gelap
        margin: 6px 0;  # Jarak atas dan bawah
    }
    .est-sub {  # Ringkasan parameter di bawah hasil
        font-size: 0.84rem;  # Ukuran teks
        color: #374151;  # Abu gelap
    }

    .stButton > button {  # Tombol aksi utama Streamlit
        background-color: #00b87c !important;  # Warna hijau utama
        color: #ffffff !important;  # Teks putih
        border: 1px solid #00b87c !important;  # Border hijau
        border-radius: 10px !important;  # Sudut membulat
        font-weight: 700 !important;  # Huruf tebal
        font-size: 0.84rem !important;  # Ukuran huruf tombol
        padding: 0.5rem 1rem !important;  # Ruang dalam tombol
    }
    .stButton > button:hover {  # Warna tombol saat kursor diarahkan
        background-color: #009e6a !important;  # Hijau sedikit lebih gelap
        border-color: #009e6a !important;  # Border menyesuaikan
    }
</style>  # Akhir blok gaya CSS
""", unsafe_allow_html=True)  # Mengizinkan render HTML/CSS di Streamlit

# ==========================================
# 2. GROQ CLIENT, MODEL & DATA CSV
# ==========================================
GROQ_API_KEY = "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"  # API key untuk akses layanan Groq LLM
client = Groq(api_key=GROQ_API_KEY)  # Membuat objek klien Groq menggunakan API key tersebut

def normalize_rupiah(val) -> int:  # Fungsi untuk menormalkan angka harga menjadi satuan rupiah penuh
    try:  # Blok percobaan konversi angka
        num = float(val)  # Mengubah nilai input menjadi tipe float
    except Exception:  # Jika gagal dikonversi ke angka
        return 0  # Kembalikan 0 sebagai nilai aman
    if 0 < num < 10000:  # Jika angka tersimpan dalam satuan ribuan (misal 600 berarti 600.000)
        num = num * 1000  # Kalikan 1000 agar menjadi nominal rupiah penuh
    return int(round(num))  # Membulatkan dan mengembalikan sebagai bilangan bulat (int)

def format_rp(val) -> str:  # Fungsi untuk memformat angka menjadi format Rupiah Indonesia (contoh: Rp 600.000)
    n = normalize_rupiah(val)  # Menormalkan nilai angka terlebih dahulu
    return f"Rp {n:,}".replace(",", ".")  # Mengganti pemisah koma ribuan menjadi titik sesuai format Indonesia

@st.cache_resource  # Cache objek model ML agar tidak dimuat ulang setiap kali halaman di-refresh
def load_model():  # Fungsi memuat model machine learning dari file model_kos.joblib
    if os.path.exists("model_kos.joblib"):  # Mengecek apakah file model_kos.joblib tersedia
        try:  # Mencoba memuat model dengan joblib
            return joblib.load("model_kos.joblib")  # Mengembalikan objek model yang sudah terlatih
        except Exception:  # Jika file rusak atau tidak kompatibel
            return None  # Kembalikan None agar aplikasi memakai formula cadangan
    return None  # Kembalikan None jika file model belum ada

@st.cache_data  # Cache hasil pembacaan dataset CSV agar proses lebih cepat
def load_data():  # Fungsi memuat dan membersihkan data kos dari kos_madiun.csv
    if not os.path.exists("kos_madiun.csv"):  # Jika file CSV tidak ditemukan
        return pd.DataFrame({  # Mengembalikan DataFrame contoh bawaan
            "Nama_Kos": [  # Daftar nama kos contoh
                "Kos Taman Kartoharjo",
                "Griya Mahasiswa Demangan",
                "Kos Asri Nambangan Kidul",
                "Wisma Melati Manguharjo",
                "Kos Nyaman Mojorejo"
            ],
            "Jarak_Kampus_km": [0.5, 1.1, 2.3, 1.6, 3.0],  # Jarak ke kampus (km)
            "Rating": [4.8, 4.5, 4.2, 4.4, 4.0],  # Rating kos
            "AC": [1, 1, 0, 1, 0],  # Status AC (1 = ada, 0 = tidak ada)
            "WiFi": [1, 1, 1, 1, 1],  # Status WiFi (1 = ada, 0 = tidak ada)
            "Harga_Bulanan": [1150000, 950000, 600000, 850000, 520000]  # Harga sewa per bulan
        }), False  # Mengembalikan False menandakan bukan dari file CSV asli

    df = pd.read_csv("kos_madiun.csv")  # Membaca file kos_madiun.csv ke DataFrame pandas
    df.columns = [str(c).strip() for c in df.columns]  # Membersihkan spasi berlebih di nama kolom

    rename_map = {}  # Kamus pemetaan nama kolom alternatif ke nama standar
    lower_cols = {c.lower(): c for c in df.columns}  # Pemetaan nama kolom versi huruf kecil

    if "Nama_Kos" not in df.columns:  # Jika kolom Nama_Kos belum ada, cari nama alternatifnya
        for k in ["nama_kos", "nama", "kos", "judul", "name"]:  # Daftar kandidat nama kolom
            if k in lower_cols:  # Jika cocok
                rename_map[lower_cols[k]] = "Nama_Kos"  # Petakan ke Nama_Kos
                break  # Hentikan pencarian untuk kolom ini

    if "Jarak_Kampus_km" not in df.columns:  # Mencari kolom jarak ke kampus jika belum bernama Jarak_Kampus_km
        for k in ["jarak_kampus_km", "jarak_km", "jarak", "distance"]:  # Kandidat nama kolom jarak
            if k in lower_cols:  # Jika ditemukan
                rename_map[lower_cols[k]] = "Jarak_Kampus_km"  # Petakan ke Jarak_Kampus_km
                break  # Keluar dari loop

    if "Rating" not in df.columns:  # Mencari kolom rating jika belum bernama Rating
        for k in ["rating", "bintang", "score"]:  # Kandidat nama kolom rating
            if k in lower_cols:  # Jika ditemukan
                rename_map[lower_cols[k]] = "Rating"  # Petakan ke Rating
                break  # Keluar dari loop

    if "AC" not in df.columns:  # Mencari kolom fasilitas AC jika belum bernama AC
        for k in ["ac", "fasilitas_ac", "has_ac"]:  # Kandidat nama kolom AC
            if k in lower_cols:  # Jika ditemukan
                rename_map[lower_cols[k]] = "AC"  # Petakan ke AC
                break  # Keluar dari loop

    if "WiFi" not in df.columns:  # Mencari kolom fasilitas WiFi jika belum bernama WiFi
        for k in ["wifi", "wi_fi", "internet", "fasilitas_wifi"]:  # Kandidat nama kolom WiFi
            if k in lower_cols:  # Jika ditemukan
                rename_map[lower_cols[k]] = "WiFi"  # Petakan ke WiFi
                break  # Keluar dari loop

    if "Harga_Bulanan" not in df.columns:  # Mencari kolom harga sewa bulanan jika belum bernama Harga_Bulanan
        for k in ["harga_bulanan", "harga", "sewa", "tarif", "price"]:  # Kandidat nama kolom harga
            if k in lower_cols:  # Jika ditemukan
                rename_map[lower_cols[k]] = "Harga_Bulanan"  # Petakan ke Harga_Bulanan
                break  # Keluar dari loop

    if rename_map:  # Jika ada kolom yang perlu diganti namanya
        df = df.rename(columns=rename_map)  # Terapkan perubahan nama kolom ke DataFrame

    for col in ["AC", "WiFi"]:  # Standarisasi nilai kolom fasilitas menjadi 1 (ada) atau 0 (tidak ada)
        if col in df.columns:  # Pastikan kolom tersedia di DataFrame
            df[col] = df[col].apply(  # Terapkan fungsi konversi ke setiap baris kolom
                lambda v: 1 if str(v).strip().lower() in ["1", "true", "ya", "y", "ada", "yes"] else 0  # Nilai 1 jika positif, selain itu 0
            )

    if "Harga_Bulanan" in df.columns:  # Membersihkan dan menormalkan kolom harga bulanan
        df["Harga_Bulanan"] = (  # Ubah menjadi string lalu buang karakter selain angka, titik, dan minus
            df["Harga_Bulanan"]
            .astype(str)
            .str.replace(r"[^0-9.-]", "", regex=True)
        )
        df["Harga_Bulanan"] = pd.to_numeric(df["Harga_Bulanan"], errors="coerce").fillna(0)  # Ubah ke numerik, isi NaN dengan 0
        df["Harga_Bulanan"] = df["Harga_Bulanan"].apply(normalize_rupiah)  # Normalkan angka ribuan menjadi rupiah penuh

    if "Jarak_Kampus_km" in df.columns:  # Pastikan kolom jarak bertipe numerik
        df["Jarak_Kampus_km"] = pd.to_numeric(df["Jarak_Kampus_km"], errors="coerce")

    if "Rating" in df.columns:  # Pastikan kolom rating bertipe numerik
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    return df, True  # Kembalikan DataFrame bersih dan status True (berhasil memuat CSV)

model = load_model()  # Memanggil fungsi load_model() saat aplikasi berjalan
df_kos, csv_loaded = load_data()  # Memanggil fungsi load_data() untuk mendapatkan data kos

# ==========================================
# 3. BASIS PENGETAHUAN FAQ + RAG MINI
# ==========================================
FAQ_KOS_MADIUN = [  # Daftar pertanyaan & jawaban umum seputar kos di Madiun (basis pengetahuan RAG)
    {
        "topik": "Kisaran Harga Bulanan",  # Judul topik FAQ
        "kunci": ["harga", "sewa", "tarif", "biaya", "bulanan", "murah", "mahal", "budget"],  # Kata kunci pencocokan
        "jawab": "Kisaran harga kos di Madiun umumnya berada di rentang Rp 500.000-an untuk kamar ekonomis hingga Rp 1.000.000+ per bulan untuk kamar dengan fasilitas lebih lengkap atau lokasi dekat kampus/pusat kota."  # Isi jawaban FAQ
    },
    {
        "topik": "Pengaruh Jarak ke Kampus",
        "kunci": ["jarak", "kampus", "dekat", "jauh", "lokasi", "strategis", "jalan"],
        "jawab": "Kos yang lebih dekat ke kampus atau pusat kota biasanya memiliki tarif lebih tinggi karena menghemat waktu dan ongkos perjalanan harian, sedangkan kos agak jauh cenderung lebih hemat."
    },
    {
        "topik": "Kamar AC vs Non-AC",
        "kunci": ["ac", "panas", "kipas", "dingin", "listrik", "token"],
        "jawab": "Kamar ber-AC memberi kenyamanan suhu saat siang/malam panas, tetapi perlu mempertimbangkan tambahan biaya sewa dan pemakaian listrik bulanan dibanding kamar non-AC."
    },
    {
        "topik": "Ketersediaan WiFi & Internet",
        "kunci": ["wifi", "internet", "kuliah", "tugas", "daring", "kuota"],
        "jawab": "Untuk mahasiswa yang sering mengerjakan tugas atau kelas daring, pastikan mengecek kestabilan sinyal WiFi kos dan apakah biayanya sudah termasuk sewa kamar."
    },
    {
        "topik": "Checklist Survei Kamar",
        "kunci": ["survei", "cek", "lihat", "kamar", "tips", "sebelum", "dp"],
        "jawab": "Saat survei kos, periksa ventilasi dan pencahayaan kamar, aliran air kamar mandi, kondisi stopkontak/listrik, keamanan parkir, serta aturan jam malam atau tamu."
    },
    {
        "topik": "Biaya Tambahan Listrik & Air",
        "kunci": ["listrik", "air", "pdam", "token", "tambahan", "iuran"],
        "jawab": "Tanyakan secara jelas apakah tarif bulanan sudah termasuk air dan listrik, atau menggunakan meteran/token terpisah agar anggaran bulanan tidak meleset."
    },
    {
        "topik": "Keamanan & Akses Lingkungan",
        "kunci": ["aman", "keamanan", "pagar", "cctv", "banjir", "akses", "gang"],
        "jawab": "Perhatikan keamanan lingkungan seperti pagar/gerbang, penerangan jalan sekitar kos, area parkir motor, dan akses jalan masuk terutama saat hujan."
    },
    {
        "topik": "Membaca Rating & Ulasan",
        "kunci": ["rating", "ulasan", "review", "bintang", "penghuni", "nyaman"],
        "jawab": "Gunakan rating sebagai gambaran awal kualitas layanan dan kenyamanan kos, lalu cocokkan dengan kebutuhan pribadi seperti ketenangan belajar atau kedekatan akses."
    },
    {
        "topik": "Pembayaran & Perjanjian Sewa",
        "kunci": ["bayar", "dp", "deposit", "kontrak", "bulanan", "perjanjian", "keluar"],
        "jawab": "Sebelum membayar DP atau sewa pertama, pastikan ada kesepakatan jelas mengenai tanggal jatuh tempo, aturan pindah/berhenti sewa, dan kebijakan pengembalian uang muka."
    }
]

def ambil_konteks_rag(pertanyaan: str, df: pd.DataFrame, faqs: list) -> str:  # Fungsi mengambil konteks FAQ & data CSV relevan (Retrieval)
    kata_tanya = set(pertanyaan.lower().replace("?", " ").replace(",", " ").split())  # Memecah pertanyaan menjadi kumpulan kata kunci huruf kecil
    skor_faq = []  # Menyimpan pasangan (skor_kecocokan, item_faq)
    for item in faqs:  # Loop setiap topik FAQ
        cocok = sum(1 for k in item["kunci"] if k in kata_tanya or any(k in w for w in kata_tanya))  # Menghitung jumlah kata kunci yang cocok
        skor_faq.append((cocok, item))  # Menambahkan skor ke daftar
    skor_faq.sort(key=lambda x: x[0], reverse=True)  # Mengurutkan dari skor tertinggi ke terendah
    terpilih = [it for sc, it in skor_faq if sc > 0][:4]  # Mengambil maksimal 4 FAQ paling relevan
    if not terpilih:  # Jika tidak ada kata kunci yang cocok sama sekali
        terpilih = faqs[:4]  # Ambil 4 FAQ pertama sebagai konteks bawaan

    teks_faq = "\n".join([f"- [{f['topik']}] {f['jawab']}" for f in terpilih])  # Menyusun teks FAQ terpilih

    ringkasan_csv = []  # Menyimpan ringkasan beberapa unit kos dari CSV
    if df is not None and len(df) > 0:  # Jika DataFrame tidak kosong
        for _, r in df.head(8).iterrows():  # Ambil maksimal 8 baris pertama agar konteks tetap ringkas
            nm = r.get("Nama_Kos", "-")  # Nama kos
            hrg = format_rp(r.get("Harga_Bulanan", 0))  # Harga terformat Rupiah
            jrk = r.get("Jarak_Kampus_km", "-")  # Jarak ke kampus
            rtg = r.get("Rating", "-")  # Rating kos
            ringkasan_csv.append(f"- {nm}: {hrg}/bln, jarak {jrk} km, rating {rtg}")  # Format baris ringkasan
    teks_csv = "\n".join(ringkasan_csv) if ringkasan_csv else "- Tidak ada data CSV."  # Gabungkan ringkasan CSV

    return f"FAQ TERKAIT:\n{teks_faq}\n\nDATA KOS TERDATA:\n{teks_csv}"  # Mengembalikan gabungan konteks FAQ dan data kos

# ==========================================
# 4. SISTEM PAKAR ATURAN BERANTAI
# ==========================================
ATURAN_PAKAR = [  # Basis aturan IF-THEN (jika-maka) untuk sistem pakar forward chaining
    {"jika": ["budget ketat", "prioritas hemat"], "maka": "profil hemat"},
    {"jika": ["dekat kampus", "prioritas lokasi"], "maka": "profil komuter kampus"},
    {"jika": ["butuh ac", "butuh wifi"], "maka": "kebutuhan fasilitas lengkap"},
    {"jika": ["profil hemat", "butuh ac"], "maka": "konflik budget ac"},
    {"jika": ["profil hemat", "tidak butuh ac"], "maka": "cocok kos ekonomis"},
    {"jika": ["profil komuter kampus", "butuh wifi"], "maka": "cocok kos mahasiswa strategis"},
    {"jika": ["kebutuhan fasilitas lengkap", "budget longgar"], "maka": "cocok kos eksklusif"},
    {"jika": ["konflik budget ac"], "maka": "REKOMENDASI: Pilih kos non-AC yang berventilasi baik atau naikkan budget ke rentang menengah."},
    {"jika": ["cocok kos ekonomis"], "maka": "REKOMENDASI: Prioritaskan kos tarif hemat dengan akses jalan aman dan air lancar."},
    {"jika": ["cocok kos mahasiswa strategis"], "maka": "REKOMENDASI: Pilih kos dekat kampus dengan koneksi WiFi stabil untuk menunjang kuliah/tugas."},
    {"jika": ["cocok kos eksklusif"], "maka": "REKOMENDASI: Pilih kos berfasilitas AC + WiFi dengan rating tinggi dan keamanan memadai."}
]

def forward_chaining_kos(aturan: list, fakta_awal: list):  # Mesin inferensi forward chaining (dari fakta menuju kesimpulan)
    fakta = list(fakta_awal)  # Menyalin daftar fakta awal agar tidak mengubah list asli secara langsung
    jejak = []  # Menyimpan urutan aturan yang berhasil terpicu
    ada_baru = True  # Penanda apakah ada fakta baru yang dihasilkan pada iterasi terakhir
    while ada_baru:  # Ulangi selama masih ada fakta baru yang bisa disimpulkan
        ada_baru = False  # Reset penanda sebelum memeriksa semua aturan
        for r in aturan:  # Periksa setiap aturan satu per satu
            terpenuhi = all(k in fakta for k in r["jika"])  # Cek apakah semua syarat pada "jika" ada di daftar fakta
            if terpenuhi and r["maka"] not in fakta:  # Jika terpenuhi dan kesimpulannya belum ada di fakta
                fakta.append(r["maka"])  # Tambahkan kesimpulan ke daftar fakta baru
                jejak.append((r["jika"], r["maka"]))  # Catat jejak aturan yang aktif
                ada_baru = True  # Tandai bahwa ada fakta baru sehingga perlu cek ulang
    rekomendasi = [f for f in fakta if f.startswith("REKOMENDASI:")]  # Filter hanya kesimpulan akhir yang diawali REKOMENDASI:
    if not rekomendasi:  # Jika belum ada rekomendasi spesifik yang muncul
        rekomendasi = ["REKOMENDASI: Kombinasikan jarak ke kampus, fasilitas wajib, dan anggaran bulanan saat survei lokasi."]  # Saran umum
    return fakta, jejak, rekomendasi  # Kembalikan semua fakta, jejak aturan, dan rekomendasi akhir

# ==========================================
# 5. SIDEBAR & HERO HEADER
# ==========================================
MENU_KOS = {  # Kamus daftar menu halaman beserta judul dan deskripsi banner-nya
    "Katalog Kos": ("Temukan Kos di Madiun", "Jelajahi unit kos, saring fasilitas AC/WiFi, dan bandingkan harga sewa bulanan."),
    "Prediksi Harga": ("Kalkulator Harga Kos", "Perkirakan tarif sewa bulanan berdasarkan jarak kampus, rating, dan fasilitas."),
    "Asisten FAQ & RAG": ("Asisten Info Kos", "Tanya jawab seputar aturan sewa, survei kamar, dan data kos Madiun."),
    "Rekomendasi Pakar": ("Rekomendasi Sistem Pakar", "Cocokkan anggaran dan kebutuhanmu lewat aturan penalaran berantai.")
}

with st.sidebar:  # Memulai blok elemen di sidebar kiri
    st.markdown(  # Menampilkan logo dan nama brand di sidebar
        '<div class="sb-brand">'
        + BRAND_SVG
        + '<div>'
        '<div class="sb-name">MadiunKos</div>'
        '<div class="sb-tag">Cari &amp; Prediksi Kos</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="sb-sec">Menu Utama</div>', unsafe_allow_html=True)  # Label bagian menu
    selected_tab = st.radio(  # Komponen radio navigasi antar halaman
        "Menu Utama",
        list(MENU_KOS.keys()),  # Pilihan menu diambil dari kunci MENU_KOS
        label_visibility="collapsed"  # Sembunyikan label bawaan radio karena sudah ada judul di atasnya
    )

    sumber_info = "kos_madiun.csv" if csv_loaded else "Data Contoh"  # Menentukan teks sumber data aktif
    model_info = "Model ML Aktif" if model is not None else "Estimasi Tarif"  # Menentukan teks status model prediksi
    st.markdown(  # Menampilkan kotak status ringkas di bawah menu sidebar
        f'<div class="sb-box">'
        f'<div><b>Wilayah:</b> Kota Madiun</div>'
        f'<div><b>Dataset:</b> {sumber_info} ({len(df_kos)} unit)</div>'
        f'<div><b>Prediksi:</b> {model_info}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

hero_title, hero_sub = MENU_KOS[selected_tab]  # Mengambil judul dan subjudul sesuai menu yang sedang dipilih
st.markdown(  # Menampilkan banner hero hijau di bagian atas konten utama
    f'<div class="app-hero">'
    f'<div>'
    f'<div class="app-hero-title">{hero_title}</div>'
    f'<div class="app-hero-sub">{hero_sub}</div>'
    f'</div>'
    f'<div class="hero-pill">Kota Madiun · Bulanan</div>'
    f'</div>',
    unsafe_allow_html=True
)

if not csv_loaded:  # Memberi informasi jika file CSV belum terbaca
    st.info("Berkas `kos_madiun.csv` belum ditemukan di folder proyek; menampilkan data contoh.")

# Menghitung nilai statistik ringkas (rata-rata, termurah, tertinggi) dari kolom Harga_Bulanan
avg_p = format_rp(df_kos["Harga_Bulanan"].mean()) if "Harga_Bulanan" in df_kos.columns and len(df_kos) > 0 else "Rp 800.000"
min_p = format_rp(df_kos["Harga_Bulanan"].min()) if "Harga_Bulanan" in df_kos.columns and len(df_kos) > 0 else "Rp 500.000"
max_p = format_rp(df_kos["Harga_Bulanan"].max()) if "Harga_Bulanan" in df_kos.columns and len(df_kos) > 0 else "Rp 1.200.000"

m1, m2, m3, m4 = st.columns(4)  # Membagi baris metrik menjadi 4 kolom sejajar
m1.markdown(f'<div class="stat-tile"><div class="stat-k">Unit Tersedia</div><div class="stat-v">{len(df_kos)} Kos</div></div>', unsafe_allow_html=True)  # Kolom 1: jumlah unit kos
m2.markdown(f'<div class="stat-tile"><div class="stat-k">Rata-rata / Bulan</div><div class="stat-v">{avg_p}</div></div>', unsafe_allow_html=True)  # Kolom 2: rata-rata harga
m3.markdown(f'<div class="stat-tile"><div class="stat-k">Mulai Dari</div><div class="stat-v">{min_p}</div></div>', unsafe_allow_html=True)  # Kolom 3: harga terendah
m4.markdown(f'<div class="stat-tile"><div class="stat-k">Tertinggi / Bulan</div><div class="stat-v">{max_p}</div></div>', unsafe_allow_html=True)  # Kolom 4: harga tertinggi

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)  # Memberi jarak vertikal sebelum konten halaman

# ==========================================
# 6. KATALOG KOS (GRID 2 KOLOM + FILTER)
# ==========================================
if selected_tab == "Katalog Kos":  # Jika pengguna memilih menu Katalog Kos
    f1, f2, f3 = st.columns([2.2, 1, 1])  # Membagi area filter menjadi 3 kolom dengan lebar proporsional
    with f1:  # Kolom 1: input pencarian nama kos
        cari_nama = st.text_input("Cari nama kos", placeholder="Misal: Melati, Mawar, Kartoharjo...")
    with f2:  # Kolom 2: dropdown filter fasilitas AC
        filter_ac = st.selectbox("Fasilitas AC", ["Semua", "Ada AC", "Non-AC"])
    with f3:  # Kolom 3: dropdown filter fasilitas WiFi
        filter_wifi = st.selectbox("Fasilitas WiFi", ["Semua", "Ada WiFi"])

    df_tampil = df_kos.copy()  # Salin DataFrame utama agar data asli tidak berubah
    if cari_nama and "Nama_Kos" in df_tampil.columns:  # Filter berdasarkan teks pencarian nama kos
        df_tampil = df_tampil[df_tampil["Nama_Kos"].astype(str).str.contains(cari_nama, case=False, na=False)]
    if filter_ac == "Ada AC" and "AC" in df_tampil.columns:  # Filter hanya kos yang memiliki AC
        df_tampil = df_tampil[df_tampil["AC"] == 1]
    elif filter_ac == "Non-AC" and "AC" in df_tampil.columns:  # Filter hanya kos tanpa AC
        df_tampil = df_tampil[df_tampil["AC"] == 0]
    if filter_wifi == "Ada WiFi" and "WiFi" in df_tampil.columns:  # Filter hanya kos yang memiliki WiFi
        df_tampil = df_tampil[df_tampil["WiFi"] == 1]

    st.markdown(f"<div style='font-weight:700; color:#008a5c; margin:6px 0 10px 0;'>Menampilkan {len(df_tampil)} unit kos</div>", unsafe_allow_html=True)  # Info jumlah unit hasil filter

    if len(df_tampil) == 0:  # Jika tidak ada data yang cocok dengan filter
        st.warning("Tidak ada unit kos yang sesuai dengan filter.")
    else:  # Jika ada unit kos yang ditampilkan
        rows_list = list(df_tampil.iterrows())  # Mengubah baris DataFrame menjadi list agar mudah dipasangkan per 2 kartu
        for idx in range(0, len(rows_list), 2):  # Loop melompat 2 baris setiap iterasi untuk grid 2 kolom
            cols = st.columns(2, gap="medium")  # Membuat 2 kolom sejajar
            for c_idx in range(2):  # Mengisi kolom kiri (0) dan kolom kanan (1)
                if idx + c_idx < len(rows_list):  # Pastikan indeks tidak melebihi jumlah data
                    _, row = rows_list[idx + c_idx]  # Mengambil data satu unit kos
                    nama = row.get("Nama_Kos", "Kos Madiun")  # Nama kos
                    harga_str = format_rp(row.get("Harga_Bulanan", 0))  # Harga terformat Rupiah
                    jarak_k = row.get("Jarak_Kampus_km", "-")  # Jarak ke kampus
                    rating_k = row.get("Rating", "-")  # Rating kos
                    ada_ac = row.get("AC", 0) == 1  # Boolean status AC
                    ada_wifi = row.get("WiFi", 0) == 1  # Boolean status WiFi

                    ac_cls = "p-badge yes" if ada_ac else "p-badge"  # Kelas badge AC
                    wifi_cls = "p-badge yes" if ada_wifi else "p-badge"  # Kelas badge WiFi
                    ac_lbl = "✓ AC" if ada_ac else "Non-AC"  # Label teks AC
                    wifi_lbl = "✓ WiFi" if ada_wifi else "Tanpa WiFi"  # Label teks WiFi
                    tipe_kamar = "Kamar AC" if ada_ac else "Kamar Hemat"  # Label tipe kamar di pojok kanan bawah

                    card_html = (  # Menyusun HTML kartu unit kos
                        '<div class="prop-card">'
                        '<div class="prop-banner"></div>'
                        '<div class="prop-body">'
                        '<div class="prop-top">'
                        f'<div class="prop-title">{nama}</div>'
                        f'<div class="prop-rating">★ {rating_k} / 5.0</div>'
                        '</div>'
                        f'<div class="prop-loc">📍 {jarak_k} km dari kampus</div>'
                        f'<div class="prop-badges"><span class="{ac_cls}">{ac_lbl}</span><span class="{wifi_cls}">{wifi_lbl}</span></div>'
                        '<div class="prop-foot">'
                        f'<div class="prop-price">{harga_str} <span>/ bulan</span></div>'
                        f'<div class="prop-tag-city">{tipe_kamar}</div>'
                        '</div>'
                        '</div>'
                        '</div>'
                    )
                    cols[c_idx].markdown(card_html, unsafe_allow_html=True)  # Render kartu di kolom yang sesuai

# ==========================================
# 7. PREDIKSI HARGA + GRAFIK
# ==========================================
elif selected_tab == "Prediksi Harga":  # Jika pengguna memilih menu Prediksi Harga
    col_l, col_r = st.columns([1.05, 1], gap="large")  # Membagi halaman menjadi 2 kolom: input (kiri) dan hasil (kanan)
    with col_l:  # Kolom kiri: pengaturan input fitur kos
        st.markdown('<div class="sec-card"><div class="sec-head">Kriteria Kamar</div><div class="sec-sub">Atur jarak kampus, rating, dan fasilitas kamar.</div>', unsafe_allow_html=True)
        jarak = st.slider("Jarak ke kampus (km)", 0.1, 10.0, 1.2, 0.1)  # Slider input jarak kampus
        rating = st.slider("Rating kos", 1.0, 5.0, 4.5, 0.1)  # Slider input rating kos
        c_a, c_b = st.columns(2)  # Membagi toggle fasilitas menjadi 2 kolom
        with c_a:  # Toggle fasilitas AC
            punya_ac = st.toggle("Kamar AC", value=True)
        with c_b:  # Toggle fasilitas WiFi
            punya_wifi = st.toggle("WiFi", value=True)
        st.button("Hitung Estimasi")  # Tombol pemicu hitung ulang/interaksi
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:  # Kolom kanan: menampilkan hasil prediksi harga bulanan
        st.markdown('<div class="sec-card"><div class="sec-head">Estimasi Sewa Bulanan</div><div class="sec-sub">Perkiraan harga kos berdasarkan model atau data.</div>', unsafe_allow_html=True)
        ac_v = 1 if punya_ac else 0  # Konversi toggle AC ke angka 1 atau 0
        wifi_v = 1 if punya_wifi else 0  # Konversi toggle WiFi ke angka 1 atau 0
        est = None  # Variabel penampung hasil prediksi
        if model is not None:  # Jika model ML (.joblib) tersedia
            try:  # Melakukan prediksi menggunakan model ML
                fitur = pd.DataFrame([[jarak, rating, ac_v, wifi_v]], columns=["Jarak_Kampus_km", "Rating", "AC", "WiFi"])  # Menyiapkan DataFrame fitur sesuai urutan training
                est = float(model.predict(fitur)[0])  # Mengambil nilai prediksi pertama
            except Exception:  # Jika terjadi error saat prediksi model
                est = None  # Set None agar masuk ke formula cadangan
        if est is None:  # Formula cadangan jika model tidak tersedia atau gagal dipanggil
            est = max(350000, 550000 + (rating * 110000) - (jarak * 45000) + (ac_v * 320000) + (wifi_v * 90000))

        est_rp = format_rp(est)  # Format hasil estimasi menjadi Rupiah
        box_est = (  # Menyusun kartu hasil estimasi harga
            '<div class="est-highlight">'
            '<div class="est-lbl">Perkiraan Biaya Kos / Bulan</div>'
            f'<div class="est-val">{est_rp}</div>'
            f'<div class="est-sub">Jarak {jarak:.1f} km · Rating {rating:.1f} · {"AC" if punya_ac else "Non-AC"} · {"WiFi" if punya_wifi else "Tanpa WiFi"}</div>'
            '</div>'
        )
        st.markdown(box_est, unsafe_allow_html=True)  # Menampilkan kotak hasil prediksi
        st.markdown('</div>', unsafe_allow_html=True)

    if "Harga_Bulanan" in df_kos.columns and "Nama_Kos" in df_kos.columns:  # Menampilkan grafik batang perbandingan harga kos
        st.markdown('<div class="sec-card"><div class="sec-head">Grafik Tarif Unit Kos</div><div class="sec-sub">Perbandingan harga bulanan pada kos_madiun.csv.</div>', unsafe_allow_html=True)
        st.bar_chart(df_kos.set_index("Nama_Kos")[["Harga_Bulanan"]], color="#00b87c")  # Render bar chart berwarna hijau Bibit
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 8. ASISTEN FAQ & RAG
# ==========================================
elif selected_tab == "Asisten FAQ & RAG":  # Jika pengguna memilih menu Asisten FAQ & RAG
    st.markdown('<div class="sec-card"><div class="sec-head">Asisten Tanya Jawab Kos Madiun</div><div class="sec-sub">Menjawab pertanyaan berdasarkan FAQ kos dan data kos_madiun.csv.</div>', unsafe_allow_html=True)

    with st.expander(f"Daftar FAQ Acuan ({len(FAQ_KOS_MADIUN)} topik)"):  # Panel lipat untuk melihat seluruh daftar FAQ acuan
        for idx, item in enumerate(FAQ_KOS_MADIUN, start=1):  # Menampilkan tiap topik FAQ bernomor
            st.markdown(f"**{idx}. {item['topik']}** — {item['jawab']}")

    if "app_kos_chat" not in st.session_state:  # Inisialisasi riwayat chat di session_state jika belum ada
        st.session_state.app_kos_chat = [
            {"role": "assistant", "content": "Halo! Ada yang ingin kamu tanyakan soal harga kos, tips survei kamar, fasilitas AC/WiFi, atau pilihan kos di Madiun?"}
        ]

    b1, b2, b3, b4 = st.columns([1.2, 1.2, 1.2, 0.7])  # Baris tombol pertanyaan cepat & reset percakapan
    prompt_click = None  # Menampung pertanyaan jika tombol cepat diklik
    if b1.button("Kisaran harga kos"):  # Tombol cepat 1
        prompt_click = "Bagaimana kisaran harga kos di Madiun dan pengaruh jarak ke kampus?"
    if b2.button("Checklist survei"):  # Tombol cepat 2
        prompt_click = "Apa saja yang perlu dicek saat survei kamar kos sebelum bayar DP?"
    if b3.button("AC & listrik bulanan"):  # Tombol cepat 3
        prompt_click = "Apa pertimbangan memilih kamar AC dan biaya listrik bulanan?"
    if b4.button("Reset"):  # Tombol reset riwayat obrolan
        st.session_state.app_kos_chat = [
            {"role": "assistant", "content": "Percakapan direset. Silakan tanya lagi."}
        ]
        st.rerun()  # Memuat ulang halaman agar chat langsung kosong/ter-reset

    for m in st.session_state.app_kos_chat:  # Menampilkan seluruh riwayat pesan chat
        with st.chat_message(m["role"]):  # Bubble chat sesuai peran (user / assistant)
            st.markdown(m["content"])

    ketikan = st.chat_input("Tanya seputar kos Madiun...")  # Kotak input teks chat di bagian bawah
    pesan = prompt_click or ketikan  # Memilih pesan dari klik tombol cepat atau ketikan user

    if pesan:  # Jika ada pertanyaan baru yang dikirim
        st.session_state.app_kos_chat.append({"role": "user", "content": pesan})  # Simpan pesan user ke riwayat
        with st.chat_message("user"):  # Tampilkan pesan user
            st.markdown(pesan)

        konteks_rag = ambil_konteks_rag(pesan, df_kos, FAQ_KOS_MADIUN)  # Ambil konteks FAQ & data CSV terkait pertanyaan

        with st.chat_message("assistant"):  # Bubble balasan dari asisten AI
            with st.spinner("Menyusun jawaban..."):  # Indikator loading saat menunggu respons LLM
                balasan = None  # Variabel hasil jawaban LLM
                err_terakhir = None  # Variabel penyimpan error jika pemanggilan gagal

                daftar_model = []  # Daftar kandidat model Groq yang tersedia
                try:  # Mengambil daftar model aktif dari akun Groq
                    for item in client.models.list().data:  # Loop setiap model yang dikembalikan API
                        mid = getattr(item, "id", "")  # Ambil ID model
                        if mid and not any(k in mid.lower() for k in ["whisper", "guard", "tts", "embed"]):  # Kecualikan model non-chat/audio/embedding
                            daftar_model.append(mid)  # Tambahkan ke daftar model chat
                except Exception as e:  # Jika gagal mengambil daftar model
                    err_terakhir = e

                for cadangan in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:  # Menambahkan model cadangan jika belum ada di daftar
                    if cadangan not in daftar_model:
                        daftar_model.append(cadangan)

                system_prompt = (  # Instruksi sistem (persona & aturan grounding RAG) untuk LLM
                    "Kamu adalah asisten informasi kos Madiun. Jawab dalam bahasa Indonesia yang jelas, ramah, dan ringkas. "
                    "Utamakan informasi dari KONTEKS di bawah ini. Jika di luar konteks, katakan secara jujur bahwa informasi tersebut belum tercantum di data/FAQ."
                )

                for nama_m in daftar_model[:6]:  # Mencoba memanggil maksimal 6 model secara berurutan (fallback)
                    try:  # Request chat completion ke Groq
                        hasil = client.chat.completions.create(
                            model=nama_m,  # Nama model yang dicoba
                            messages=[  # Pesan sistem + konteks RAG + pertanyaan pengguna
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"KONTEKS:\n{konteks_rag}\n\nPERTANYAAN:\n{pesan}"}
                            ],
                            temperature=0.4,  # Kreativitas terukur agar jawaban tetap fokus pada konteks
                            max_tokens=550  # Batas panjang maksimum jawaban
                        )
                        balasan = hasil.choices[0].message.content  # Ambil teks jawaban dari respons API
                        break  # Keluar dari loop jika sudah berhasil mendapat balasan
                    except Exception as e:  # Jika model ini gagal/error, simpan error dan coba model berikutnya
                        err_terakhir = e
                        continue

                if balasan is None:  # Jika seluruh percobaan model gagal
                    balasan = f"Gagal memanggil model LLM. Detail: `{err_terakhir}`"

                st.markdown(balasan)  # Tampilkan jawaban AI di layar
                st.session_state.app_kos_chat.append({"role": "assistant", "content": balasan})  # Simpan balasan ke riwayat chat

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 9. REKOMENDASI PAKAR
# ==========================================
else:  # Jika pengguna memilih menu Rekomendasi Pakar
    p_left, p_right = st.columns([1.05, 1], gap="large")  # Membagi halaman menjadi 2 kolom: input fakta (kiri) dan hasil inferensi (kanan)
    with p_left:  # Kolom kiri: checkbox kondisi kebutuhan pengguna
        st.markdown('<div class="sec-card"><div class="sec-head">Kebutuhan &amp; Anggaranmu</div><div class="sec-sub">Centang kondisi yang sesuai untuk menjalankan penalaran aturan berantai.</div>', unsafe_allow_html=True)
        f_hemat = st.checkbox("Budget bulanan ketat / hemat", value=True)  # Fakta budget ketat
        f_longgar = st.checkbox("Budget bulanan longgar (>= Rp 1 juta)")  # Fakta budget longgar
        f_dekat = st.checkbox("Ingin jarak dekat kampus")  # Fakta ingin dekat kampus
        f_lokasi = st.checkbox("Lokasi strategis prioritas utama", value=True)  # Fakta prioritas lokasi
        f_ac = st.checkbox("Perlu fasilitas AC")  # Fakta butuh AC
        f_no_ac = st.checkbox("Tidak masalah tanpa AC", value=True)  # Fakta tidak masalah tanpa AC
        f_wifi = st.checkbox("Perlu WiFi", value=True)  # Fakta butuh WiFi
        st.button("Jalankan Rekomendasi")  # Tombol pemicu proses rekomendasi
        st.markdown('</div>', unsafe_allow_html=True)

    with p_right:  # Kolom kanan: hasil rekomendasi akhir dan jejak aturan forward chaining
        st.markdown('<div class="sec-card"><div class="sec-head">Rekomendasi &amp; Jejak Aturan</div><div class="sec-sub">Penalaran berantai (forward chaining) dari 11 aturan basis pengetahuan.</div>', unsafe_allow_html=True)
        fakta_awal = []  # Menyusun daftar fakta awal berdasarkan checkbox yang dicentang
        if f_hemat:
            fakta_awal.append("budget ketat")
        if f_longgar:
            fakta_awal.append("budget longgar")
        if f_dekat:
            fakta_awal.append("dekat kampus")
        if f_lokasi:
            fakta_awal.append("prioritas lokasi")
        if f_hemat or f_lokasi:
            fakta_awal.append("prioritas hemat")
        if f_ac:
            fakta_awal.append("butuh ac")
        if f_no_ac:
            fakta_awal.append("tidak butuh ac")
        if f_wifi:
            fakta_awal.append("butuh wifi")

        _, jejak, rekomendasi = forward_chaining_kos(ATURAN_PAKAR, fakta_awal)  # Menjalankan mesin inferensi forward chaining

        for rek in rekomendasi:  # Menampilkan kartu rekomendasi akhir
            st.markdown(
                f'<div class="est-highlight">'
                f'<div class="est-lbl">Saran Pilihan Kos</div>'
                f'<div class="est-sub" style="font-size:0.95rem; font-weight:700; color:#111827; margin-top:6px;">{rek}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='height:14px'></div><div style='font-weight:700; margin-bottom:6px;'>Jejak Aturan yang Terpicu:</div>", unsafe_allow_html=True)  # Judul daftar jejak aturan
        if jejak:  # Jika ada aturan yang aktif
            for idx, (kondisi, kesimpulan) in enumerate(jejak, start=1):  # Menampilkan alur IF ... THEN ... secara berurutan
                st.markdown(f"{idx}. JIKA `{' DAN '.join(kondisi)}` MAKA **{kesimpulan}**")
        else:  # Jika tidak ada aturan yang terpicu
            st.markdown("- Belum ada aturan yang terpicu dari fakta yang dipilih.")
        st.markdown('</div>', unsafe_allow_html=True)
