import streamlit as st  # Mengimpor library Streamlit untuk membangun antarmuka web interaktif
import pandas as pd  # Mengimpor pandas untuk membaca dan mengolah data CSV dalam bentuk DataFrame
import joblib  # Mengimpor joblib untuk memuat model machine learning (.joblib)
import os  # Mengimpor os untuk memeriksa apakah file model atau dataset tersedia di direktori
from groq import Groq  # Mengimpor Groq client untuk fitur asisten tanya-jawab berbasis LLM

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(  # Mengatur konfigurasi dasar halaman Streamlit
    page_title="MadiunKos — Cari & Prediksi Hunian Madiun",  # Menentukan judul tab di browser
    page_icon="🏠",  # Menentukan ikon tab di browser
    layout="wide",  # Mengatur tata letak halaman agar melebar penuh
    initial_sidebar_state="expanded"  # Membuka sidebar di sebelah kiri saat pertama kali dibuka
)

BRAND_SVG = (  # Menyimpan kode ikon SVG berwarna hijau ala Bibit (#00B87C)
    '<svg width="38" height="38" viewBox="0 0 38 38" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="38" height="38" rx="11" fill="#00B87C"/>'
    '<path d="M11 19.2L19 12L27 19.2V27H21.2V21.5H16.8V27H11V19.2Z" fill="#FFFFFF"/>'
    '<circle cx="28.5" cy="10.5" r="3.5" fill="#B8F3DF"/>'
    '</svg>'
)

# Menyuntikkan CSS tema hijau ala Bibit, tata letak sidebar, kartu metrik, dan kartu kos 2 kolom
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    [data-testid="stDecoration"] {
        display: none;
    }

    .stApp {
        background-color: #f7fbf9;
        color: #111827;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e5efea !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2rem;
    }
    .sb-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 15px;
        margin-bottom: 15px;
        border-bottom: 1px solid #eef5f1;
    }
    .sb-name {
        font-size: 1.15rem;
        font-weight: 800;
        color: #111827;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .sb-tag {
        font-size: 0.73rem;
        font-weight: 700;
        color: #00b87c;
        margin-top: 2px;
    }
    .sb-sec {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #6b7280;
        margin-bottom: 8px;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 6px;
        background: transparent !important;
        padding: 0 !important;
        border: none !important;
        box-shadow: none !important;
        margin-bottom: 18px;
        width: 100%;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: #ffffff !important;
        border: 1px solid #e5efea !important;
        border-radius: 10px !important;
        padding: 10px 13px !important;
        cursor: pointer;
        width: 100%;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label * {
        color: #374151 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #e6f9f2 !important;
        border-color: #9ce8ce !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover * {
        color: #008a5c !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: #e6f9f2 !important;
        border-color: #00b87c !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] *,
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked),
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) * {
        color: #008a5c !important;
        font-weight: 800 !important;
    }

    .sb-box {
        background: #f4fbf8;
        border: 1px solid #d4f2e6;
        border-radius: 12px;
        padding: 13px 14px;
        font-size: 0.78rem;
        color: #374151;
        line-height: 1.65;
    }
    .sb-box b {
        color: #008a5c;
    }

    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    .app-hero {
        background: linear-gradient(120deg, #00b87c 0%, #00a36e 60%, #10b981 100%);
        border-radius: 18px;
        padding: 22px 26px;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 14px;
        margin-bottom: 16px;
        box-shadow: 0 10px 24px -12px rgba(0, 184, 124, 0.45);
    }
    .app-hero-title {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .app-hero-sub {
        font-size: 0.85rem;
        color: #e6f9f2;
        margin-top: 4px;
    }
    .hero-pill {
        background: rgba(255, 255, 255, 0.22);
        border: 1px solid rgba(255, 255, 255, 0.38);
        color: #ffffff;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 7px 14px;
        border-radius: 999px;
    }

    .stat-tile {
        background: #ffffff;
        border: 1px solid #e5efea;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 2px 6px rgba(0, 184, 124, 0.04);
    }
    .stat-k {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #6b7280;
    }
    .stat-v {
        font-size: 1.25rem;
        font-weight: 800;
        color: #111827;
        margin-top: 4px;
    }

    .sec-card {
        background: #ffffff;
        border: 1px solid #e5efea;
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(17, 24, 39, 0.03);
    }
    .sec-head {
        font-size: 1.02rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 2px;
    }
    .sec-sub {
        font-size: 0.82rem;
        color: #6b7280;
        margin-bottom: 12px;
    }

    .prop-card {
        background: #ffffff;
        border: 1px solid #e5efea;
        border-radius: 16px;
        overflow: hidden;
        margin-bottom: 14px;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .prop-card:hover {
        transform: translateY(-2px);
        border-color: #00b87c;
        box-shadow: 0 10px 20px -12px rgba(0, 184, 124, 0.25);
    }
    .prop-banner {
        height: 6px;
        background: linear-gradient(90deg, #00b87c 0%, #34d399 100%);
    }
    .prop-body {
        padding: 16px 18px;
    }
    .prop-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 10px;
    }
    .prop-title {
        font-size: 1.02rem;
        font-weight: 800;
        color: #111827;
    }
    .prop-rating {
        font-size: 0.76rem;
        font-weight: 700;
        color: #008a5c;
        background: #e6f9f2;
        border: 1px solid #b8f3df;
        padding: 3px 9px;
        border-radius: 999px;
        white-space: nowrap;
    }
    .prop-loc {
        font-size: 0.82rem;
        color: #6b7280;
        margin-top: 4px;
    }
    .prop-badges {
        margin-top: 10px;
        margin-bottom: 14px;
    }
    .p-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        color: #4b5563;
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        padding: 3px 10px;
        border-radius: 999px;
        margin-right: 6px;
    }
    .p-badge.yes {
        color: #008a5c;
        background: #e6f9f2;
        border-color: #9ce8ce;
    }
    .prop-foot {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-top: 12px;
        border-top: 1px solid #f0f5f2;
    }
    .prop-price {
        font-size: 1.08rem;
        font-weight: 800;
        color: #00b87c;
    }
    .prop-price span {
        font-size: 0.76rem;
        font-weight: 500;
        color: #6b7280;
    }
    .prop-tag-city {
        font-size: 0.72rem;
        font-weight: 700;
        color: #008a5c;
        background: #f4fbf8;
        border: 1px solid #d4f2e6;
        padding: 3px 9px;
        border-radius: 999px;
    }

    .est-highlight {
        background: linear-gradient(135deg, #f4fbf8 0%, #e6f9f2 100%);
        border: 1.5px solid #00b87c;
        border-radius: 16px;
        padding: 22px;
    }
    .est-lbl {
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #008a5c;
    }
    .est-val {
        font-size: 2.05rem;
        font-weight: 800;
        color: #111827;
        margin: 6px 0;
    }
    .est-sub {
        font-size: 0.84rem;
        color: #374151;
    }

    .stButton > button {
        background-color: #00b87c !important;
        color: #ffffff !important;
        border: 1px solid #00b87c !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        padding: 0.5rem 1rem !important;
    }
    .stButton > button:hover {
        background-color: #009e6a !important;
        border-color: #009e6a !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. GROQ CLIENT, MODEL & DATA CSV
# ==========================================
GROQ_API_KEY = "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"  # API key untuk autentikasi ke layanan Groq
client = Groq(api_key=GROQ_API_KEY)  # Membuat instance klien Groq untuk memanggil model LLM

def normalize_rupiah(val) -> int:  # Menormalkan angka ribuan (misal 600) menjadi rupiah penuh (600000)
    try:  # Mencoba mengubah input menjadi angka desimal
        num = float(val)
    except Exception:  # Jika gagal dikonversi, kembalikan 0
        return 0
    if 0 < num < 10000:  # Jika nilai di bawah 10.000, anggap satuan ribuan dan kalikan 1.000
        num = num * 1000
    return int(round(num))  # Membulatkan dan mengembalikan hasil sebagai integer

def format_rp(val) -> str:  # Mengubah angka menjadi format Rupiah dengan pemisah titik (Rp 600.000)
    n = normalize_rupiah(val)  # Menormalkan nilai angka terlebih dahulu
    return f"Rp {n:,}".replace(",", ".")  # Mengganti koma pemisah ribuan menjadi titik

@st.cache_resource  # Menyimpan objek model di cache agar tidak dimuat ulang tiap render
def load_model():  # Memuat model machine learning dari file model_kos.joblib jika tersedia
    if os.path.exists("model_kos.joblib"):  # Mengecek keberadaan file model_kos.joblib
        try:  # Mencoba memuat model menggunakan joblib
            return joblib.load("model_kos.joblib")
        except Exception:  # Jika model gagal dimuat, kembalikan None agar memakai formula fallback
            return None
    return None  # Mengembalikan None jika file model belum ada

@st.cache_data  # Menyimpan hasil pembacaan CSV di cache agar akses lebih cepat
def load_data():  # Memuat, menstandarkan nama kolom, dan membersihkan data dari kos_madiun.csv
    if not os.path.exists("kos_madiun.csv"):  # Jika file CSV belum ada, pakai data contoh internal
        return pd.DataFrame({
            "Nama_Kos": [
                "Kos Taman Kartoharjo",
                "Griya Mahasiswa Demangan",
                "Kos Asri Nambangan Kidul",
                "Wisma Melati Manguharjo",
                "Kos Nyaman Mojorejo"
            ],
            "Jarak_Kampus_km": [0.5, 1.1, 2.3, 1.6, 3.0],
            "Rating": [4.8, 4.5, 4.2, 4.4, 4.0],
            "AC": [1, 1, 0, 1, 0],
            "WiFi": [1, 1, 1, 1, 1],
            "Harga_Bulanan": [1150000, 950000, 600000, 850000, 520000]
        }), False

    df = pd.read_csv("kos_madiun.csv")  # Membaca file kos_madiun.csv ke DataFrame
    df.columns = [str(c).strip() for c in df.columns]  # Menghapus spasi berlebih pada nama kolom

    rename_map = {}  # Menyimpan pemetaan nama kolom alternatif ke nama standar
    lower_cols = {c.lower(): c for c in df.columns}  # Kamus nama kolom versi huruf kecil

    if "Nama_Kos" not in df.columns:  # Mencari kolom alternatif untuk nama kos
        for k in ["nama_kos", "nama", "kos", "judul", "name"]:
            if k in lower_cols:
                rename_map[lower_cols[k]] = "Nama_Kos"
                break

    if "Jarak_Kampus_km" not in df.columns:  # Mencari kolom alternatif untuk jarak ke kampus
        for k in ["jarak_kampus_km", "jarak_km", "jarak", "distance"]:
            if k in lower_cols:
                rename_map[lower_cols[k]] = "Jarak_Kampus_km"
                break

    if "Rating" not in df.columns:  # Mencari kolom alternatif untuk rating
        for k in ["rating", "bintang", "score"]:
            if k in lower_cols:
                rename_map[lower_cols[k]] = "Rating"
                break

    if "AC" not in df.columns:  # Mencari kolom alternatif untuk fasilitas AC
        for k in ["ac", "fasilitas_ac", "has_ac"]:
            if k in lower_cols:
                rename_map[lower_cols[k]] = "AC"
                break

    if "WiFi" not in df.columns:  # Mencari kolom alternatif untuk fasilitas WiFi
        for k in ["wifi", "wi_fi", "internet", "fasilitas_wifi"]:
            if k in lower_cols:
                rename_map[lower_cols[k]] = "WiFi"
                break

    if "Harga_Bulanan" not in df.columns:  # Mencari kolom alternatif untuk harga sewa bulanan
        for k in ["harga_bulanan", "harga", "sewa", "tarif", "price"]:
            if k in lower_cols:
                rename_map[lower_cols[k]] = "Harga_Bulanan"
                break

    if rename_map:  # Menerapkan perubahan nama kolom jika ada yang dipetakan
        df = df.rename(columns=rename_map)

    for col in ["AC", "WiFi"]:  # Menstandarkan nilai fasilitas menjadi 1 (ada) atau 0 (tidak ada)
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: 1 if str(v).strip().lower() in ["1", "true", "ya", "y", "ada", "yes"] else 0
            )

    if "Harga_Bulanan" in df.columns:  # Membersihkan karakter non-angka lalu menormalkan harga
        df["Harga_Bulanan"] = (
            df["Harga_Bulanan"]
            .astype(str)
            .str.replace(r"[^0-9.-]", "", regex=True)
        )
        df["Harga_Bulanan"] = pd.to_numeric(df["Harga_Bulanan"], errors="coerce").fillna(0)
        df["Harga_Bulanan"] = df["Harga_Bulanan"].apply(normalize_rupiah)

    if "Jarak_Kampus_km" in df.columns:  # Mengonversi kolom jarak menjadi numerik
        df["Jarak_Kampus_km"] = pd.to_numeric(df["Jarak_Kampus_km"], errors="coerce")

    if "Rating" in df.columns:  # Mengonversi kolom rating menjadi numerik
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    return df, True  # Mengembalikan DataFrame bersih dan status berhasil memuat CSV

model = load_model()  # Memuat model prediksi harga saat aplikasi berjalan
df_kos, csv_loaded = load_data()  # Memuat data kos dari CSV

# ==========================================
# 3. BASIS PENGETAHUAN FAQ + RAG MINI
# ==========================================
FAQ_KOS_MADIUN = [  # Daftar topik FAQ kos Madiun sebagai basis pengetahuan retrieval
    {
        "topik": "Kisaran Harga Bulanan",
        "kunci": ["harga", "sewa", "tarif", "biaya", "bulanan", "murah", "mahal", "budget"],
        "jawab": "Kisaran harga kos di Madiun umumnya berada di rentang Rp 500.000-an untuk kamar ekonomis hingga Rp 1.000.000+ per bulan untuk kamar dengan fasilitas lebih lengkap atau lokasi dekat kampus/pusat kota."
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

def ambil_konteks_rag(pertanyaan: str, df: pd.DataFrame, faqs: list) -> str:  # Mengambil konteks FAQ & data kos yang relevan dengan pertanyaan
    kata_tanya = set(pertanyaan.lower().replace("?", " ").replace(",", " ").split())  # Memecah pertanyaan jadi kumpulan kata
    skor_faq = []  # Menyimpan pasangan skor kecocokan dan item FAQ
    for item in faqs:  # Menghitung kecocokan kata kunci tiap topik FAQ
        cocok = sum(1 for k in item["kunci"] if k in kata_tanya or any(k in w for w in kata_tanya))
        skor_faq.append((cocok, item))
    skor_faq.sort(key=lambda x: x[0], reverse=True)  # Mengurutkan dari kecocokan tertinggi
    terpilih = [it for sc, it in skor_faq if sc > 0][:4]  # Mengambil maksimal 4 FAQ relevan
    if not terpilih:  # Jika tidak ada kata kunci yang cocok, pakai 4 FAQ pertama
        terpilih = faqs[:4]

    teks_faq = "\n".join([f"- [{f['topik']}] {f['jawab']}" for f in terpilih])  # Menyusun teks konteks FAQ

    ringkasan_csv = []  # Menyusun ringkasan data unit kos dari CSV
    if df is not None and len(df) > 0:
        for _, r in df.head(8).iterrows():
            nm = r.get("Nama_Kos", "-")
            hrg = format_rp(r.get("Harga_Bulanan", 0))
            jrk = r.get("Jarak_Kampus_km", "-")
            rtg = r.get("Rating", "-")
            ringkasan_csv.append(f"- {nm}: {hrg}/bln, jarak {jrk} km, rating {rtg}")
    teks_csv = "\n".join(ringkasan_csv) if ringkasan_csv else "- Tidak ada data CSV."

    return f"FAQ TERKAIT:\n{teks_faq}\n\nDATA KOS TERDATA:\n{teks_csv}"  # Mengembalikan gabungan konteks untuk prompt LLM

# ==========================================
# 4. SISTEM PAKAR ATURAN BERANTAI
# ==========================================
ATURAN_PAKAR = [  # Basis aturan JIKA-MAKA untuk penalaran forward chaining
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

def forward_chaining_kos(aturan: list, fakta_awal: list):  # Menjalankan inferensi maju dari fakta awal hingga menghasilkan rekomendasi
    fakta = list(fakta_awal)  # Menyalin daftar fakta awal
    jejak = []  # Menyimpan urutan aturan yang terpicu
    ada_baru = True  # Penanda iterasi selama masih ada kesimpulan baru
    while ada_baru:
        ada_baru = False
        for r in aturan:
            terpenuhi = all(k in fakta for k in r["jika"])  # Cek apakah semua premis pada "jika" terpenuhi
            if terpenuhi and r["maka"] not in fakta:  # Jika terpenuhi dan belum tercatat di fakta
                fakta.append(r["maka"])  # Tambahkan kesimpulan baru ke daftar fakta
                jejak.append((r["jika"], r["maka"]))  # Simpan jejak aturan
                ada_baru = True
    rekomendasi = [f for f in fakta if f.startswith("REKOMENDASI:")]  # Ambil kesimpulan akhir rekomendasi
    if not rekomendasi:  # Saran cadangan jika belum ada aturan akhir yang cocok
        rekomendasi = ["REKOMENDASI: Kombinasikan jarak ke kampus, fasilitas wajib, dan anggaran bulanan saat survei lokasi."]
    return fakta, jejak, rekomendasi

# ==========================================
# 5. SIDEBAR & HERO HEADER
# ==========================================
MENU_KOS = {  # Daftar menu navigasi beserta judul dan deskripsi halaman
    "Katalog Kos": ("Temukan Kos di Madiun", "Jelajahi unit kos, saring fasilitas AC/WiFi, dan bandingkan harga sewa bulanan."),
    "Prediksi Harga": ("Kalkulator Harga Kos", "Perkirakan tarif sewa bulanan berdasarkan jarak kampus, rating, dan fasilitas."),
    "Asisten FAQ & RAG": ("Asisten Info Kos", "Tanya jawab seputar aturan sewa, survei kamar, dan data kos Madiun."),
    "Rekomendasi Pakar": ("Rekomendasi Sistem Pakar", "Cocokkan anggaran dan kebutuhanmu lewat aturan penalaran berantai.")
}

with st.sidebar:  # Menyusun elemen navigasi dan informasi di sidebar kiri
    st.markdown(
        '<div class="sb-brand">'
        + BRAND_SVG
        + '<div>'
        '<div class="sb-name">MadiunKos</div>'
        '<div class="sb-tag">Cari &amp; Prediksi Kos</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="sb-sec">Menu Utama</div>', unsafe_allow_html=True)
    selected_tab = st.radio(  # Memilih halaman fitur yang sedang dibuka
        "Menu Utama",
        list(MENU_KOS.keys()),
        label_visibility="collapsed"
    )

    sumber_info = "kos_madiun.csv" if csv_loaded else "Data Contoh"  # Label status sumber data
    model_info = "Model ML Aktif" if model is not None else "Estimasi Tarif"  # Label status model prediksi
    st.markdown(
        f'<div class="sb-box">'
        f'<div><b>Wilayah:</b> Kota Madiun</div>'
        f'<div><b>Dataset:</b> {sumber_info} ({len(df_kos)} unit)</div>'
        f'<div><b>Prediksi:</b> {model_info}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

hero_title, hero_sub = MENU_KOS[selected_tab]  # Mengambil judul dan subjudul sesuai menu aktif
st.markdown(
    f'<div class="app-hero">'
    f'<div>'
    f'<div class="app-hero-title">{hero_title}</div>'
    f'<div class="app-hero-sub">{hero_sub}</div>'
    f'</div>'
    f'<div class="hero-pill">Kota Madiun · Bulanan</div>'
    f'</div>',
    unsafe_allow_html=True
)

if not csv_loaded:  # Memberi tahu pengguna jika sedang memakai data contoh
    st.info("Berkas `kos_madiun.csv` belum ditemukan di folder proyek; menampilkan data contoh.")

# Menghitung statistik ringkas (rata-rata, terendah, tertinggi) dari harga bulanan kos
avg_p = format_rp(df_kos["Harga_Bulanan"].mean()) if "Harga_Bulanan" in df_kos.columns and len(df_kos) > 0 else "Rp 800.000"
min_p = format_rp(df_kos["Harga_Bulanan"].min()) if "Harga_Bulanan" in df_kos.columns and len(df_kos) > 0 else "Rp 500.000"
max_p = format_rp(df_kos["Harga_Bulanan"].max()) if "Harga_Bulanan" in df_kos.columns and len(df_kos) > 0 else "Rp 1.200.000"

m1, m2, m3, m4 = st.columns(4)  # Menampilkan 4 kartu statistik sejajar
m1.markdown(f'<div class="stat-tile"><div class="stat-k">Unit Tersedia</div><div class="stat-v">{len(df_kos)} Kos</div></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="stat-tile"><div class="stat-k">Rata-rata / Bulan</div><div class="stat-v">{avg_p}</div></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="stat-tile"><div class="stat-k">Mulai Dari</div><div class="stat-v">{min_p}</div></div>', unsafe_allow_html=True)
m4.markdown(f'<div class="stat-tile"><div class="stat-k">Tertinggi / Bulan</div><div class="stat-v">{max_p}</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ==========================================
# 6. KATALOG KOS (GRID 2 KOLOM + FILTER)
# ==========================================
if selected_tab == "Katalog Kos":  # Logika fitur Katalog Kos
    f1, f2, f3 = st.columns([2.2, 1, 1])  # Kolom pencarian nama dan filter AC/WiFi
    with f1:
        cari_nama = st.text_input("Cari nama kos", placeholder="Misal: Melati, Mawar, Kartoharjo...")
    with f2:
        filter_ac = st.selectbox("Fasilitas AC", ["Semua", "Ada AC", "Non-AC"])
    with f3:
        filter_wifi = st.selectbox("Fasilitas WiFi", ["Semua", "Ada WiFi"])

    df_tampil = df_kos.copy()  # Menyaring DataFrame berdasarkan filter aktif
    if cari_nama and "Nama_Kos" in df_tampil.columns:
        df_tampil = df_tampil[df_tampil["Nama_Kos"].astype(str).str.contains(cari_nama, case=False, na=False)]
    if filter_ac == "Ada AC" and "AC" in df_tampil.columns:
        df_tampil = df_tampil[df_tampil["AC"] == 1]
    elif filter_ac == "Non-AC" and "AC" in df_tampil.columns:
        df_tampil = df_tampil[df_tampil["AC"] == 0]
    if filter_wifi == "Ada WiFi" and "WiFi" in df_tampil.columns:
        df_tampil = df_tampil[df_tampil["WiFi"] == 1]

    st.markdown(f"<div style='font-weight:700; color:#008a5c; margin:6px 0 10px 0;'>Menampilkan {len(df_tampil)} unit kos</div>", unsafe_allow_html=True)

    if len(df_tampil) == 0:  # Pesan jika hasil filter kosong
        st.warning("Tidak ada unit kos yang sesuai dengan filter.")
    else:  # Menampilkan daftar kos dalam grid 2 kolom
        rows_list = list(df_tampil.iterrows())
        for idx in range(0, len(rows_list), 2):
            cols = st.columns(2, gap="medium")
            for c_idx in range(2):
                if idx + c_idx < len(rows_list):
                    _, row = rows_list[idx + c_idx]
                    nama = row.get("Nama_Kos", "Kos Madiun")
                    harga_str = format_rp(row.get("Harga_Bulanan", 0))
                    jarak_k = row.get("Jarak_Kampus_km", "-")
                    rating_k = row.get("Rating", "-")
                    ada_ac = row.get("AC", 0) == 1
                    ada_wifi = row.get("WiFi", 0) == 1

                    ac_cls = "p-badge yes" if ada_ac else "p-badge"
                    wifi_cls = "p-badge yes" if ada_wifi else "p-badge"
                    ac_lbl = "✓ AC" if ada_ac else "Non-AC"
                    wifi_lbl = "✓ WiFi" if ada_wifi else "Tanpa WiFi"
                    tipe_kamar = "Kamar AC" if ada_ac else "Kamar Hemat"

                    card_html = (
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
                    cols[c_idx].markdown(card_html, unsafe_allow_html=True)

# ==========================================
# 7. PREDIKSI HARGA + GRAFIK
# ==========================================
elif selected_tab == "Prediksi Harga":  # Logika fitur Prediksi Harga & grafik tarif
    col_l, col_r = st.columns([1.05, 1], gap="large")
    with col_l:  # Panel input parameter kamar
        st.markdown('<div class="sec-card"><div class="sec-head">Kriteria Kamar</div><div class="sec-sub">Atur jarak kampus, rating, dan fasilitas kamar.</div>', unsafe_allow_html=True)
        jarak = st.slider("Jarak ke kampus (km)", 0.1, 10.0, 1.2, 0.1)
        rating = st.slider("Rating kos", 1.0, 5.0, 4.5, 0.1)
        c_a, c_b = st.columns(2)
        with c_a:
            punya_ac = st.toggle("Kamar AC", value=True)
        with c_b:
            punya_wifi = st.toggle("WiFi", value=True)
        st.button("Hitung Estimasi")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:  # Panel hasil prediksi harga sewa bulanan
        st.markdown('<div class="sec-card"><div class="sec-head">Estimasi Sewa Bulanan</div><div class="sec-sub">Perkiraan harga kos berdasarkan model atau data.</div>', unsafe_allow_html=True)
        ac_v = 1 if punya_ac else 0
        wifi_v = 1 if punya_wifi else 0
        est = None
        if model is not None:  # Menggunakan model ML jika berhasil dimuat
            try:
                fitur = pd.DataFrame([[jarak, rating, ac_v, wifi_v]], columns=["Jarak_Kampus_km", "Rating", "AC", "WiFi"])
                est = float(model.predict(fitur)[0])
            except Exception:
                est = None
        if est is None:  # Formula cadangan jika model tidak tersedia
            est = max(350000, 550000 + (rating * 110000) - (jarak * 45000) + (ac_v * 320000) + (wifi_v * 90000))

        est_rp = format_rp(est)
        box_est = (
            '<div class="est-highlight">'
            '<div class="est-lbl">Perkiraan Biaya Kos / Bulan</div>'
            f'<div class="est-val">{est_rp}</div>'
            f'<div class="est-sub">Jarak {jarak:.1f} km · Rating {rating:.1f} · {"AC" if punya_ac else "Non-AC"} · {"WiFi" if punya_wifi else "Tanpa WiFi"}</div>'
            '</div>'
        )
        st.markdown(box_est, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if "Harga_Bulanan" in df_kos.columns and "Nama_Kos" in df_kos.columns:  # Grafik perbandingan tarif kos
        st.markdown('<div class="sec-card"><div class="sec-head">Grafik Tarif Unit Kos</div><div class="sec-sub">Perbandingan harga bulanan pada kos_madiun.csv.</div>', unsafe_allow_html=True)
        st.bar_chart(df_kos.set_index("Nama_Kos")[["Harga_Bulanan"]], color="#00b87c")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 8. ASISTEN FAQ & RAG
# ==========================================
elif selected_tab == "Asisten FAQ & RAG":  # Logika fitur Asisten FAQ & RAG berbasis Groq
    st.markdown('<div class="sec-card"><div class="sec-head">Asisten Tanya Jawab Kos Madiun</div><div class="sec-sub">Menjawab pertanyaan berdasarkan FAQ kos dan data kos_madiun.csv.</div>', unsafe_allow_html=True)

    with st.expander(f"Daftar FAQ Acuan ({len(FAQ_KOS_MADIUN)} topik)"):  # Menampilkan daftar FAQ yang dijadikan acuan
        for idx, item in enumerate(FAQ_KOS_MADIUN, start=1):
            st.markdown(f"**{idx}. {item['topik']}** — {item['jawab']}")

    if "app_kos_chat" not in st.session_state:  # Menginisialisasi riwayat chat di session_state
        st.session_state.app_kos_chat = [
            {"role": "assistant", "content": "Halo! Ada yang ingin kamu tanyakan soal harga kos, tips survei kamar, fasilitas AC/WiFi, atau pilihan kos di Madiun?"}
        ]

    b1, b2, b3, b4 = st.columns([1.2, 1.2, 1.2, 0.7])  # Tombol pertanyaan cepat dan reset obrolan
    prompt_click = None
    if b1.button("Kisaran harga kos"):
        prompt_click = "Bagaimana kisaran harga kos di Madiun dan pengaruh jarak ke kampus?"
    if b2.button("Checklist survei"):
        prompt_click = "Apa saja yang perlu dicek saat survei kamar kos sebelum bayar DP?"
    if b3.button("AC & listrik bulanan"):
        prompt_click = "Apa pertimbangan memilih kamar AC dan biaya listrik bulanan?"
    if b4.button("Reset"):
        st.session_state.app_kos_chat = [
            {"role": "assistant", "content": "Percakapan direset. Silakan tanya lagi."}
        ]
        st.rerun()

    for m in st.session_state.app_kos_chat:  # Merender seluruh riwayat pesan chat
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    ketikan = st.chat_input("Tanya seputar kos Madiun...")  # Input pertanyaan dari pengguna
    pesan = prompt_click or ketikan

    if pesan:  # Mengirim konteks RAG + pertanyaan ke model Groq
        st.session_state.app_kos_chat.append({"role": "user", "content": pesan})
        with st.chat_message("user"):
            st.markdown(pesan)

        konteks_rag = ambil_konteks_rag(pesan, df_kos, FAQ_KOS_MADIUN)

        with st.chat_message("assistant"):
            with st.spinner("Menyusun jawaban..."):
                balasan = None
                err_terakhir = None

                daftar_model = []  # Menyusun daftar kandidat model chat Groq
                try:
                    for item in client.models.list().data:
                        mid = getattr(item, "id", "")
                        if mid and not any(k in mid.lower() for k in ["whisper", "guard", "tts", "embed"]):
                            daftar_model.append(mid)
                except Exception as e:
                    err_terakhir = e

                for cadangan in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:  # Model fallback cadangan
                    if cadangan not in daftar_model:
                        daftar_model.append(cadangan)

                system_prompt = (
                    "Kamu adalah asisten informasi kos Madiun. Jawab dalam bahasa Indonesia yang jelas, ramah, dan ringkas. "
                    "Utamakan informasi dari KONTEKS di bawah ini. Jika di luar konteks, katakan secara jujur bahwa informasi tersebut belum tercantum di data/FAQ."
                )

                for nama_m in daftar_model[:6]:  # Mencoba model secara berurutan sampai berhasil
                    try:
                        hasil = client.chat.completions.create(
                            model=nama_m,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"KONTEKS:\n{konteks_rag}\n\nPERTANYAAN:\n{pesan}"}
                            ],
                            temperature=0.4,
                            max_tokens=550
                        )
                        balasan = hasil.choices[0].message.content
                        break
                    except Exception as e:
                        err_terakhir = e
                        continue

                if balasan is None:
                    balasan = f"Gagal memanggil model LLM. Detail: `{err_terakhir}`"

                st.markdown(balasan)
                st.session_state.app_kos_chat.append({"role": "assistant", "content": balasan})

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 9. REKOMENDASI PAKAR
# ==========================================
else:  # Logika fitur Rekomendasi Pakar (forward chaining)
    p_left, p_right = st.columns([1.05, 1], gap="large")
    with p_left:  # Checkbox fakta kebutuhan dan anggaran pengguna
        st.markdown('<div class="sec-card"><div class="sec-head">Kebutuhan &amp; Anggaranmu</div><div class="sec-sub">Centang kondisi yang sesuai untuk menjalankan penalaran aturan berantai.</div>', unsafe_allow_html=True)
        f_hemat = st.checkbox("Budget bulanan ketat / hemat", value=True)
        f_longgar = st.checkbox("Budget bulanan longgar (>= Rp 1 juta)")
        f_dekat = st.checkbox("Ingin jarak dekat kampus")
        f_lokasi = st.checkbox("Lokasi strategis prioritas utama", value=True)
        f_ac = st.checkbox("Perlu fasilitas AC")
        f_no_ac = st.checkbox("Tidak masalah tanpa AC", value=True)
        f_wifi = st.checkbox("Perlu WiFi", value=True)
        st.button("Jalankan Rekomendasi")
        st.markdown('</div>', unsafe_allow_html=True)

    with p_right:  # Menampilkan hasil rekomendasi dan jejak aturan yang aktif
        st.markdown('<div class="sec-card"><div class="sec-head">Rekomendasi &amp; Jejak Aturan</div><div class="sec-sub">Penalaran berantai (forward chaining) dari 11 aturan basis pengetahuan.</div>', unsafe_allow_html=True)
        fakta_awal = []
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

        _, jejak, rekomendasi = forward_chaining_kos(ATURAN_PAKAR, fakta_awal)

        for rek in rekomendasi:
            st.markdown(
                f'<div class="est-highlight">'
                f'<div class="est-lbl">Saran Pilihan Kos</div>'
                f'<div class="est-sub" style="font-size:0.95rem; font-weight:700; color:#111827; margin-top:6px;">{rek}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='height:14px'></div><div style='font-weight:700; margin-bottom:6px;'>Jejak Aturan yang Terpicu:</div>", unsafe_allow_html=True)
        if jejak:
            for idx, (kondisi, kesimpulan) in enumerate(jejak, start=1):
                st.markdown(f"{idx}. JIKA `{' DAN '.join(kondisi)}` MAKA **{kesimpulan}**")
        else:
            st.markdown("- Belum ada aturan yang terpicu dari fakta yang dipilih.")
        st.markdown('</div>', unsafe_allow_html=True)
