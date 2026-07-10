import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Konfigurasi Halaman
st.set_page_config(page_title="Customer Retention Prediction", layout="centered")

st.title("🛒 Prediksi Retensi Pelanggan")
st.write("Aplikasi ini memprediksi apakah pelanggan akan kembali berbelanja (*Retention*) atau tidak (*Churn*) berdasarkan data historis interaksi mereka.")

# Memuat model, scaler, dan fitur
try:
    with open('rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('X_columns.pkl', 'rb') as file:
        X_columns = pickle.load(file)
except FileNotFoundError:
    st.error("⚠️ File model (.pkl) tidak ditemukan. Harap jalankan Jupyter Notebook terlebih dahulu sampai ke tahap Deployment untuk men-generate file model.")
    st.stop()

# Layout Form Input
st.subheader("Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Umur (Tahun)", min_value=15, max_value=80, value=25)
    pages_viewed = st.number_input("Jumlah Halaman Dilihat", min_value=1, max_value=100, value=15)
    duration = st.number_input("Durasi Akses Web/App (Menit)", min_value=1.0, max_value=200.0, value=20.0)
    delivery_time = st.number_input("Waktu Pengiriman Pesanan (Hari)", min_value=1, max_value=30, value=3)

with col2:
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    city = st.selectbox("Kota Domisili", ["Jakarta", "Surabaya", "Bandung", "Medan"])
    category = st.selectbox("Kategori Produk Favorit", ["Elektronik", "Fashion", "Makanan", "Kesehatan"])

# Tombol Prediksi
if st.button("🔍 Prediksi Status Pelanggan", use_container_width=True):
    # Kumpulkan Data
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'City': [city],
        'Product_Category': [category],
        'Pages_Viewed': [pages_viewed],
        'Duration_Mins': [duration],
        'Delivery_Time_Days': [delivery_time]
    })
    
    # 1. Encoding
    input_encoded = pd.get_dummies(input_data)
    
    # 2. Sesuaikan Kolom (Pastikan sama persis dengan saat training)
    for col in X_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
            
    input_encoded = input_encoded[X_columns] # Re-order
    
    # 3. Scaling Numerikal
    numerical_cols = ['Age', 'Pages_Viewed', 'Duration_Mins', 'Delivery_Time_Days']
    input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])
    
    # 4. Prediksi
    prediction = model.predict(input_encoded)[0]
    
    # 5. Output Hasil
    st.markdown("---")
    if prediction == 1:
        st.success("### ✅ Prediksi: RETENTION")
        st.write("Pelanggan ini berpotensi tinggi untuk **kembali berbelanja**. Pertahankan layanan Anda!")
    else:
        st.error("### ❌ Prediksi: CHURN")
        st.write("Pelanggan ini berisiko **tidak akan kembali**. Pertimbangkan untuk memberikan intervensi khusus seperti promo gratis ongkir atau voucher diskon.")
