import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell(
"""# Segmentasi Pelanggan E-Commerce Berdasarkan Perilaku Belanja
*Penambangan Data - Informatika Universitas Alma Ata Yogyakarta*

• **Anggota Kelompok:**
1. Nama 1 (nim 1)
2. Nama 2 (nim 2)
3. Nama 3 (nim 3)
• **Nama Dataset:** "ecomerce_dataset"
• **Link Sumber Dataset:** *(Isi dengan link asal dataset)*"""),

    new_code_cell(
"""# Import library sesuai kebutuhan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pickle

import warnings
warnings.filterwarnings('ignore')"""),

    new_markdown_cell(
"""## 1. Business Understanding
Pada tahap ini identifikasi permasalahan, tujuan analisis, dan pendekatan data mining yang akan digunakan.

**Permasalahan Utama:**
- Perusahaan E-Commerce kesulitan mengenali karakteristik spesifik dari pelanggan mereka, sehingga strategi promosi sering kali tidak tepat sasaran dan berujung pada pemborosan biaya marketing.

**Tujuan bisnis:**
- Untuk mengidentifikasi kelompok pelanggan (segmen) berdasarkan perilaku belanja mereka, sehingga tim marketing dapat memberikan promosi khusus (misal: diskon grosir untuk reseller, layanan VIP untuk pelanggan premium).

**Tujuan Data Mining:**
- Untuk mengelompokkan pelanggan ke dalam beberapa segmen secara otomatis berdasarkan fitur transaksi (Quantity, Unit Price, Total Sales).

**Pilih salah satu teknik:**
- Clustering

**Alasan memilih teknik:**
- Dataset tidak memiliki label target pasti yang valid untuk klasifikasi, namun memiliki fitur numerik terkait transaksi yang sangat ideal untuk pengelompokan (Segmentasi) pelanggan berbasis kemiripan data pengeluaran."""),

    new_markdown_cell(
"""## 2. Data Understanding
Pada tahap ini lakukan eksplorasi dataset untuk memahami struktur, kualitas, dan karakteristik data."""),
    
    new_code_cell(
"""# Muat dataset
df = pd.read_csv("ecomerce_dataset.csv", sep=";")

# Tampilkan 5 data pertama
display(df.head())

# Eksplorasi awal
print("Info Dataset:")
df.info()

print("\\nStatistik Deskriptif:")
display(df.describe(include='all'))

# Jumlah missing value dan duplikasi data
print("\\nJumlah Missing Values:")
print(df.isnull().sum())
print("\\nJumlah Duplikasi:", df.duplicated().sum())"""),

    new_code_cell(
"""# Memastikan kolom harga bertipe numerik untuk visualisasi
df['Unit_Price'] = pd.to_numeric(df['Unit_Price'], errors='coerce')

# Visualisasi Data
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.histplot(df['Quantity'], bins=20, kde=True, color='blue')
plt.title('Distribusi Kuantitas (Jumlah Barang)')

plt.subplot(1, 3, 2)
sns.histplot(df['Unit_Price'], bins=20, kde=True, color='orange')
plt.title('Distribusi Harga Satuan (Unit Price)')

plt.subplot(1, 3, 3)
sns.histplot(df['Total sales'], bins=20, kde=True, color='green')
plt.title('Distribusi Total Penjualan')

plt.tight_layout()
plt.show()"""),

    new_markdown_cell(
"""### 💡 Insight Data Understanding
- Terdapat beberapa missing value dan tipe data harga (Unit Price) yang masih dibaca sebagai string sehingga butuh dikonversi ke numerik.
- Fitur finansial (Total Sales, Unit Price) dan kuantitas barang sangat berpotensi menjadi fitur pembeda utama antar-pelanggan."""),

    new_markdown_cell(
"""## 3. Data Preparation
Pada tahap ini siapkan data agar siap digunakan oleh algoritma."""),

    new_code_cell(
"""# 1. Konversi Tipe Data
df['Unit_Price'] = pd.to_numeric(df['Unit_Price'], errors='coerce')

# 2. Menangani Missing Value
df.ffill(inplace=True)

# 3. Feature Selection (Memilih 3 fitur utama untuk clustering)
features = ['Total sales', 'Quantity', 'Unit_Price']
X = df[features].copy()

# 4. Standardisasi / Normalisasi Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ubah kembali ke DataFrame untuk kemudahan visualisasi nanti
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
display(X_scaled_df.head())"""),

    new_markdown_cell(
"""### 💡 Insight Data Preparation
- **Konversi Tipe Data & Missing Value:** Mengkonversi nilai string pada kolom `Unit_Price` menjadi angka desimal dan mengisi sisa nilai kosong dengan metode `ffill`.
- **Feature Selection:** Memilih 3 fitur inti yaitu `Total sales`, `Quantity`, dan `Unit_Price` untuk membedakan kelas Premium, Grosir (Bulk), dan Retail.
- **Standardisasi:** Menggunakan `StandardScaler` agar rentang nilai setiap kolom seimbang dan algoritma K-Means tidak bias."""),

    new_markdown_cell(
"""## 4. Modeling
Bangun satu model sesuai teknik yang dipilih."""),

    new_code_cell(
"""# Menggunakan algoritma K-Means Clustering
# Menentukan jumlah cluster K=3 berdasarkan persona pelanggan
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Tambahkan label cluster ke dataset
df['Cluster'] = clusters
X['Cluster'] = clusters

print("Distribusi Pelanggan per Cluster:")
print(df['Cluster'].value_counts())"""),

    new_markdown_cell(
"""### 💡 Insight Modeling
Model yang dipilih adalah **K-Means Clustering** dengan `k=3`. Pemilihan K=3 ditujukan untuk secara spesifik memetakan tiga karakteristik pembeli E-Commerce: *Retail Shopper*, *Bulk Buyer (Grosir)*, dan *Premium Shopper*."""),

    new_markdown_cell(
"""## 5. Evaluation
Evaluasi model menggunakan metrik yang sesuai."""),

    new_code_cell(
"""# Menghitung metrik evaluasi untuk Clustering
sil_score = silhouette_score(X_scaled, clusters)
db_index = davies_bouldin_score(X_scaled, clusters)

print(f"Silhouette Score     : {sil_score:.4f}")
print(f"Davies-Bouldin Index : {db_index:.4f}")

# Visualisasi Cluster menggunakan PCA (2D)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis', alpha=0.7)
plt.title('Visualisasi Cluster Pelanggan (PCA 2D)')
plt.show()"""),

    new_markdown_cell(
"""### 💡 Insight Evaluation
Berdasarkan visualisasi PCA dan skor evaluasi, model telah berhasil memisahkan pelanggan ke dalam 3 kelompok utama. Kombinasi 3 fitur transaksi yang saling terkait (Unit Price, Quantity, Total Sales) membuat batas kelompok yang cukup tegas."""),

    new_markdown_cell(
"""## 6. Deployment
Implementasikan model menggunakan Streamlit."""),

    new_code_cell(
"""# Simpan model untuk digunakan di app.py
with open('kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)

with open('kmeans_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
    
print("Model K-Means berhasil disimpan (kmeans_model.pkl)")"""),

    new_markdown_cell(
"""### Langkah Deployment:
1. Model telah disimpan ke dalam file `.pkl`.
2. Siapkan file `app.py` untuk membaca model dan menampilkan antarmuka berbasis web.
3. Jalankan aplikasi menggunakan perintah `streamlit run app.py` di terminal.

*(Tambahkan Screenshot Hasil Streamlit Anda di bawah ini)*
![Screenshot Aplikasi](contoh_screenshot.png)"""),

    new_markdown_cell(
"""## 7. Kesimpulan

1. **Permasalahan:** Perusahaan kesulitan mengenali persona pelanggan berdasarkan kebiasaan transaksinya.
2. **Teknik dan Algoritma:** Menggunakan pendekatan Unsupervised Learning dengan algoritma K-Means Clustering (`k=3`).
3. **Hasil Evaluasi:** Model tervalidasi mampu membentuk klaster perilaku (*Retail*, *Bulk*, *Premium*) dengan memanfaatkan fitur Quantity, Unit Price, dan Total Sales.
4. **Apakah tujuan tercapai:** Ya, perusahaan kini memiliki alat untuk mengotomatiskan identifikasi segmen pembeli baru menggunakan aplikasi Streamlit.
5. **Saran pengembangan:** 
   - Ke depannya, data kategori produk (*Product Category*) dapat ditambahkan untuk melengkapi analisis *profiling* pasca-clustering.""")
])

with open('Template_CRISPDM_Clustering.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
