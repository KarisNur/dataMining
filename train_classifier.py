import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import joblib

def run_antigravity_classification_pipeline(data_path="ecomerce_dataset.xlsx"):
    print("[INFO] Initializing Antigravity Classification Pipeline...")
    
    # 1. Load Dataset
    df = pd.read_excel(data_path, sheet_name="E_Commerce_Dataset")
    
    # 2. Feature Engineering (Temporal Extraction)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Order_Month'] = df['Date'].dt.month
    df['Order_DayOfWeek'] = df['Date'].dt.dayofweek
    
    # 3. Define Explicit Target and Feature Spaces
    target_column = 'Is_Returning_Customer'
    
    categorical_features = ['Gender', 'City', 'Product_Category', 'Payment_Method', 'Device_Type']
    numeric_features = ['Age', 'Unit_Price', 'Quantity', 'Total sales', 'Discount sales', 
                        'Discount_Percentage', 'Session_Duration_Minutes', 'Pages_Viewed', 
                        'Order_Month', 'Order_DayOfWeek']
    
    # Isolate X and y (Exclude identifiers and post-purchase variables to avoid leakage)
    X = df[categorical_features + numeric_features]
    y = df[target_column].astype(int)  # Convert boolean True/False to 1/0
    
    # 4. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Training Features Shape : {X_train.shape}")
    print(f"[INFO] Testing Features Shape  : {X_test.shape}")
    print(f"[INFO] Class Distribution      : Train Positive={y_train.sum()}, Train Negative={len(y_train)-y_train.sum()}")
    
    import os
    os.makedirs('data', exist_ok=True)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data.to_csv('data/train_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)
    print("[INFO] Exported training and testing datasets to data/ directory.")
    
    # 5. Pipeline Preprocessing Stages
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # 6. Model Definition (Naive Bayes Classifier)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])
    
    # 7. Model Training Execution
    print("[INFO] Training Classifier Model...")
    model_pipeline.fit(X_train, y_train)
    
    # 8. Inference and Comprehensive Evaluation
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n================ CLASSIFICATION PERFORMANCE SUMMARY ================")
    print(f"Accuracy Score : {accuracy:.4f}")
    print(f"ROC-AUC Score  : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"   Predicted Neg  Predicted Pos")
    print(f"Actual Neg: {cm[0][0]:<14} {cm[0][1]}")
    print(f"Actual Pos: {cm[1][0]:<14} {cm[1][1]}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['New Customer (0)', 'Returning Customer (1)']))
    print("====================================================================\n")
    
    # 9. Model Exporting
    print("[INFO] Exporting model to models/antigravity_customer_retention_classifier.pkl")
    joblib.dump(model_pipeline, 'models/antigravity_customer_retention_classifier.pkl')
    print("[INFO] Export complete.")
    
    return model_pipeline

if __name__ == "__main__":
    trained_classifier = run_antigravity_classification_pipeline("ecomerce_dataset.xlsx")
