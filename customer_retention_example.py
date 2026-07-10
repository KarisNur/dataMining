import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=== Pipeline Klasifikasi Customer Retention ===")
    
    # 1. PERSIAPAN DATA (MOCK DATA)
    # Di sini kita membuat mock data yang sesuai dengan skenario Anda.
    # Jika Anda ingin menggunakan file excel yang ada di folder (misal: ecomerce_dataset.xlsx), 
    # Anda bisa menggantinya dengan: df = pd.read_excel('ecomerce_dataset.xlsx')
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(18, 65, n_samples),
        'Gender': np.random.choice(['Laki-laki', 'Perempuan'], n_samples),
        'City': np.random.choice(['Jakarta', 'Surabaya', 'Bandung', 'Medan'], n_samples),
        'Product_Category': np.random.choice(['Elektronik', 'Fashion', 'Makanan', 'Kesehatan'], n_samples),
        'Pages_Viewed': np.random.randint(1, 50, n_samples),
        'Duration_Mins': np.random.uniform(1.0, 60.0, n_samples),
        'Delivery_Time_Days': np.random.randint(1, 14, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Membuat target variable buatan (Logika bisnis sederhana)
    # Pelanggan lebih mungkin kembali jika pengiriman cepat dan engagement (pages viewed/duration) tinggi
    score = (df['Pages_Viewed'] * 0.5) + (df['Duration_Mins'] * 0.2) - (df['Delivery_Time_Days'] * 2)
    # Mengubah score menjadi binary (1: Kembali, 0: Tidak)
    df['Is_Returning_Customer'] = (score > score.median()).astype(int)
    
    print("\n[INFO] Sampel Data (5 baris pertama):")
    print(df.head())
    print("\n[INFO] Distribusi Target (Is_Returning_Customer):")
    print(df['Is_Returning_Customer'].value_counts())

    # 2. PREPROCESSING DATA
    # Pisahkan fitur (X) dan target (y)
    X = df.drop('Is_Returning_Customer', axis=1)
    y = df['Is_Returning_Customer']

    # Encoding variabel kategorikal (Gender, City, Product_Category)
    categorical_cols = ['Gender', 'City', 'Product_Category']
    # Kita menggunakan One-Hot Encoding melalui pd.get_dummies
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Split data menjadi Training dan Testing (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Scaling fitur numerikal agar skalanya seragam
    scaler = StandardScaler()
    # Identifikasi kolom numerik (kolom yang tidak termasuk dummy variables)
    numerical_cols = ['Age', 'Pages_Viewed', 'Duration_Mins', 'Delivery_Time_Days']
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # 3. PEMODELAN (TRAINING)
    # Menggunakan Random Forest Classifier
    print("\n[INFO] Melatih Model Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # 4. EVALUASI MODEL
    y_pred = rf_model.predict(X_test)
    
    print("\n=== HASIL EVALUASI ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Menampilkan Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Customer Retention')
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("[INFO] Confusion matrix disimpan sebagai 'confusion_matrix.png'")

    # 5. INSIGHTS / FEATURE IMPORTANCE
    # Mencari tahu faktor apa yang paling berpengaruh
    feature_importances = pd.DataFrame(
        {'Feature': X_train.columns, 'Importance': rf_model.feature_importances_}
    ).sort_values(by='Importance', ascending=False)

    print("\n=== FEATURE IMPORTANCE (Faktor Paling Berpengaruh) ===")
    print(feature_importances.head(10))

    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title('Faktor Paling Berpengaruh terhadap Retensi Pelanggan')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("[INFO] Grafik feature importance disimpan sebagai 'feature_importance.png'")

if __name__ == "__main__":
    main()
