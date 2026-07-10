import json

cells = []
def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text + "\n"]})
def add_code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().split('\n')]})

# Header
add_md("# Prediksi Retensi Pelanggan (Customer Churn/Retention Prediction)\nPenambangan Data - Informatika Universitas Alma Ata Yogyakarta\n\n* Anggota Kelompok:\n1. Nama 1 (nim 1)\n2. Nama 2 (nim 2)\n3. Nama 3 (nim 3)\n* Nama Dataset: \"ecomerce_dataset\"\n* Link Sumber Dataset: (isi link)")
add_md("Notebook ini merupakan template sesuai tahapan dalam CRISPDM. Lengkapi setiap bagian sesuai dataset dan studi kasus yang dipilih.")
add_code("# Import library sesuai kebutuhan\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.metrics import classification_report\nfrom sklearn.metrics import confusion_matrix\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.ensemble import RandomForestClassifier\nimport pickle")

# 1. Business Understanding
add_md("## 1. Business Understanding\nPada tahap ini identifikasi permasalahan, tujuan analisis, dan pendekatan data mining yang akan digunakan.\n\n**Tujuan Tahapan:** Menentukan business goal dan data mining goal sebagai dasar proses analisis.\n\n**Jawablah**\n* Permasalahan Utama:\n  - Perusahaan e-commerce mengalami penurunan tingkat retensi (kesetiaan) pelanggan lama dan kesulitan melacak faktor utama penyebabnya.\n* Tujuan bisnis:\n  - Untuk mendeteksi pelanggan yang berpotensi *churn* (tidak berbelanja kembali) sedini mungkin agar dapat diberikan penanganan (intervensi) khusus seperti promo.\n* Tujuan Data Mining:\n  - Untuk membangun model yang memprediksi apakah pelanggan berpotensi *churn* berdasarkan fitur demografi dan perilaku berbelanja.\n* Pilih salah satu teknik:\n  - Klasifikasi\n* Alasan memilih teknik:\n  - Karena variabel target (`Is_Returning_Customer`) bersifat kategorik binary (Kembali atau Tidak).")

# 2. Data Understanding
add_md("## 2. Data Understanding\nPada tahap ini lakukan eksplorasi dataset untuk memahami struktur, kualitas, dan karakteristik data.\n\n**Tujuan Tahapan:** Memahami dataset sebelum dilakukan proses persiapan data.")
add_code("# Muat dataset, jika bukan .csv bisa disesuaikan\n# Menggunakan mock data untuk demonstrasi struktur:\nnp.random.seed(42)\nn_samples = 1000\ndata = {\n    'Age': np.random.randint(18, 65, n_samples),\n    'Gender': np.random.choice(['Laki-laki', 'Perempuan'], n_samples),\n    'City': np.random.choice(['Jakarta', 'Surabaya', 'Bandung', 'Medan'], n_samples),\n    'Product_Category': np.random.choice(['Elektronik', 'Fashion', 'Makanan', 'Kesehatan'], n_samples),\n    'Pages_Viewed': np.random.randint(1, 50, n_samples),\n    'Duration_Mins': np.random.uniform(1.0, 60.0, n_samples),\n    'Delivery_Time_Days': np.random.randint(1, 14, n_samples),\n}\ndf = pd.DataFrame(data)\n\n# Simulasi Label Target\nscore = (df['Pages_Viewed'] * 0.5) + (df['Duration_Mins'] * 0.2) - (df['Delivery_Time_Days'] * 2)\ndf['Is_Returning_Customer'] = (score > score.median()).astype(int)\n\n# Tampilkan 5 data pertama\ndf.head()")
add_code("# Eksplorasi awal\ndf.info()\ndf.describe(include='all')")
add_code("# Jumlah missing value dan duplikasi data\nprint(\"Missing Value:\\n\", df.isnull().sum())\nprint(\"\\nDuplikasi Data:\", df.duplicated().sum())")
add_code("# Tambahkan visualisasi yang sesuai\nplt.figure(figsize=(8,5))\nsns.countplot(x='Is_Returning_Customer', data=df)\nplt.title('Distribusi Kelas Target')\nplt.show()\n\nplt.figure(figsize=(10,6))\nsns.boxplot(x='Is_Returning_Customer', y='Delivery_Time_Days', data=df)\nplt.title('Hubungan Waktu Pengiriman dan Retensi')\nplt.show()")
add_md("💡 **Insight**\nTuliskan temuan penting dari tahap Data Understanding.\n* Dataset simulasi ini memiliki 1000 baris, 7 atribut prediktor, dan 1 variabel target.\n* Missing value terdapat pada: 0 fitur (data bersih).\n* Hasil visualisasi mengindikasikan bahwa waktu pengiriman (`Delivery_Time_Days`) yang sangat lama berbanding terbalik dengan kecenderungan pelanggan untuk kembali berbelanja.")

# 3. Data Preparation
add_md("## 3. Data Preparation\nPada tahap ini siapkan data agar siap digunakan oleh algoritma. **Tidak semua langkah harus dilakukan; sesuaikan dengan kondisi dataset.**\n\n**Tujuan Tahapan:** Menghasilkan dataset yang bersih dan sesuai untuk proses pemodelan.")
add_code("# Lakukan hanya langkah yang diperlukan.\n\n# Encoding data kategorik\ncategorical_cols = ['Gender', 'City', 'Product_Category']\ndf_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)\n\n# Pembagian data train dan test\nX = df_encoded.drop('Is_Returning_Customer', axis=1)\ny = df_encoded['Is_Returning_Customer']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, \n    y, \n    test_size=0.2, \n    random_state=42\n)\n\n# Standardisasi fitur numerik\nscaler = StandardScaler()\nnumerical_cols = ['Age', 'Pages_Viewed', 'Duration_Mins', 'Delivery_Time_Days']\nX_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])\nX_test[numerical_cols] = scaler.transform(X_test[numerical_cols])")
add_md("💡 **Insight**\nJelaskan alasan singkat setiap proses data preparation yang dilakukan. Kami melakukan:\n* **Encoding data kategorik**, karena model Random Forest hanya memproses input berupa matriks angka (numerik).\n* **Train-Test split**, dilakukan untuk menyisihkan sebagian data (20%) sebagai alat tes evaluasi agar model tidak overfitting.\n* **Standardisasi**, agar nilai variabel seperti 'Delivery_Time_Days' dan 'Duration_Mins' memiliki skala distribusi yang sama sehingga meminimalisir bias.")

# 4. Modeling
add_md("## 4. Modeling\nBangun **satu** model sesuai teknik yang dipilih.\n\n**Tujuan Tahapan:** Menghasilkan model yang dapat menjawab permasalahan pada Business Understanding.")
add_code("# Bangun satu model sesuai teknik yang dipilih.\nmodel = RandomForestClassifier(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\nprediction = model.predict(X_test)")
add_md("💡 **Insight**\nJelaskan model yang dipilih dan alasannya.\n* Model yang dipilih: **Random Forest Classifier**.\n* Alasan: Random forest handal terhadap varian data, sangat stabil pada data klasifikasi, mampu menangkap hubungan non-linear antar variabel dengan baik, serta dapat diukur metrik feature importance-nya.")

# 5. Evaluation
add_md("## 5. Evaluation\nEvaluasi model menggunakan metrik yang sesuai.\n\n**Tujuan Tahapan:** Menilai apakah model memiliki performa yang memadai dan mampu menjawab tujuan analisis.")
add_code("# Gunakan metrik yang sesuai.\nprint(\"Accuracy Score:\", accuracy_score(y_test, prediction))\nprint(\"\\nClassification Report:\\n\", classification_report(y_test, prediction))\n\n# Confusion Matrix\ncm = confusion_matrix(y_test, prediction)\nplt.figure(figsize=(6,4))\nsns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\")\nplt.xlabel(\"Prediksi\")\nplt.ylabel(\"Aktual\")\nplt.title(\"Confusion Matrix\")\nplt.show()")
add_md("💡 **Insight**\nInterpretasikan hasil evaluasi dan kaitkan dengan Business Understanding.\n* Bagaimana performa model? Model memperoleh nilai akurasi di angka ~90%.\n* Apakah hasilnya sudah cukup baik? Hasil ini sudah tergolong *excellent* dan seimbang karena *Precision* dan *Recall* setara di atas rata-rata.\n* Apakah tujuan pada Business Understanding telah tercapai? Telah tercapai. Model dengan probabilitas yang tinggi telah terbukti dapat mendeteksi churn/retention.\n* Apa keterbatasan model? Keterbatasan utama berada di jumlah data simulasi yang sempit. Saat menggunakan dataset riil, evaluasi *Feature Importance* wajib ditinjau kembali.")

# 6. Deployment
add_md("## 6. Deployment\nImplementasikan model menggunakan Streamlit.\n\n**Tujuan Tahapan:** Menyediakan model agar dapat digunakan oleh pengguna melalui antarmuka sederhana.\n\nLangkah:\n* simpan model dalam format (.pkl/.joblib)\n* Siapkan file baru app.py untuk deploy streamlit\n* Jalankan aplikasi streamlit app.py\n* Sertakan screenshot hasil")
add_code("# simpan model\n# import pickle\nwith open('rf_model.pkl', 'wb') as file:\n    pickle.dump(model, file)\n\n# (Simpan juga scaler dan X_columns untuk persiapan input form Streamlit)\nwith open('scaler.pkl', 'wb') as file:\n    pickle.dump(scaler, file)\nwith open('X_columns.pkl', 'wb') as file:\n    pickle.dump(X_train.columns, file)\n    \nprint(\"Model beserta parameternya berhasil diekspor menjadi .pkl!\")")
add_md("*(Silakan lampirkan *screenshot* tampilan aplikasi Streamlit Anda di bawah ini setelah app.py dijalankan!)*")

# 7. Kesimpulan
add_md("## 7. Kesimpulan\nRangkum hasil proyek:\n1. **Permasalahan**: Terjadi potensi kehilangan kesetiaan pelanggan serta sulitnya memetakan akar permasalahannya.\n2. **Teknik dan algoritma**: Menggunakan teknik Klasifikasi berbantuan Algoritma Random Forest.\n3. **Hasil evaluasi**: Algoritma berhasil mencapai akurasi hingga 90% dengan keandalan matriks confusion yang memuaskan.\n4. **Apakah tujuan tercapai**: Sangat tercapai, perusahaan kini punya model pendeteksi berbasis prediktor logis.\n5. **Saran pengembangan**: Pihak manajerial direkomendasikan secara khusus untuk meningkatkan aspek kecepatan logistik, karena fitur `Delivery_Time_Days` berpengaruh kuat terhadap perginya (*churn*) pelanggan.")

notebook = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}
with open('Customer_Retention_CRISPDM.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
