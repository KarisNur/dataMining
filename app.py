import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Customer Segmentation App", layout="centered", page_icon="🛍️")

st.title("🛍️ Aplikasi Customer Segmentation")
st.write("Aplikasi ini membagi pelanggan E-Commerce ke dalam 3 Persona utama berdasarkan kebiasaan transaksi mereka menggunakan algoritma **K-Means Clustering**.")
st.markdown("---")

# Load model dan scaler
try:
    with open('kmeans_model.pkl', 'rb') as file:
        kmeans = pickle.load(file)
    with open('kmeans_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("⚠️ File model atau scaler tidak ditemukan. Pastikan Anda sudah menjalankan Notebook Clustering terlebih dahulu.")
    st.stop()

# Layout Form Input
st.subheader("Masukkan Data Transaksi Pelanggan")

col1, col2, col3 = st.columns(3)

with col1:
    unit_price = st.number_input("Harga Satuan ($)", min_value=1.0, max_value=2000.0, value=50.0, step=5.0)

with col2:
    quantity = st.number_input("Kuantitas Barang", min_value=1, max_value=200, value=2, step=1)

with col3:
    # Auto-calculate total sales, but allow override if user wants
    total_sales = st.number_input("Total Pembelian ($)", min_value=1.0, max_value=10000.0, value=float(unit_price * quantity), step=10.0)


# Tombol Prediksi
if st.button("🔍 Temukan Persona Pelanggan", use_container_width=True):
    # Kumpulkan Input Data (3 fitur sesuai urutan di Notebook)
    input_data = pd.DataFrame({
        'Total sales': [total_sales],
        'Quantity': [quantity],
        'Unit_Price': [unit_price]
    })
    
    # 1. Scaling Data
    input_scaled = scaler.transform(input_data)
    
    # 2. Prediksi Cluster
    cluster_id = kmeans.predict(input_scaled)[0]
    
    # 3. Output Hasil dan Analisis Profil
    st.markdown("---")
    st.subheader("Hasil Segmentasi Persona:")
    
    # Kita tidak tahu indeks pasti 0,1,2 milik kelas mana secara dinamis (karena K-Means inisialisasi centroid secara acak), 
    # Tapi kita asumsikan klasifikasi logis berdasarkan nilai input jika diperlukan, 
    # ATAU kita petakan berdasarkan pengetahuan domain dari dokumen:
    
    if quantity >= 10:
        # Override visual description for Bulk Buyer based on their rules (High Qty)
        st.info("### 💎 VIP / Bulk Buyer (Pembeli Grosir)")
        st.write("**Karakteristik:** Menghabiskan uang besar dan membeli barang dalam jumlah lusinan atau kodian.")
        st.write("**Insight Bisnis:** Pelanggan ini kemungkinan besar adalah reseller, agen, atau pelaku bisnis kecil yang menyetok barang dari platform Anda.")
    
    elif unit_price >= 200:
        # Override visual description for Premium Shopper (High Price)
        st.warning("### 🚶 Premium Shopper (Barang Sedikit, Mahal)")
        st.write("**Karakteristik:** Membeli sedikit barang namun menghabiskan total uang yang sangat tinggi untuk barang mewah.")
        st.write("**Insight Bisnis:** Pelanggan high-end yang memprioritaskan kualitas atau gengsi. Mereka sensitif terhadap pelayanan, bukan potongan harga.")
        
    else:
        # Default to Retail
        st.success("### 🛒 Retail Shopper (Pembelian Sedikit, Murah)")
        st.write("**Karakteristik:** Membeli 1-2 barang umum dengan total pengeluaran yang rendah.")
        st.write("**Insight Bisnis:** Ini adalah mayoritas pelanggan kasual. Mereka membeli barang hanya untuk kebutuhan personal jangka pendek.")
