# Antigravity Machine Learning Blueprint: E-Commerce Customer Retention Classification

This document provides a comprehensive data understanding analysis and a production-ready blueprint for the **antigravity** system to automatically ingest, preprocess, and train predictive machine learning classification models using the `ecomerce_dataset.xlsx` dataset.

---

## 1. Executive Summary & Workload Overview

The objective of this blueprint is to construct a binary classification model targeting customer loyalty and retention. By predicting whether a shopper is a returning customer, the platform can dynamically tailor marketing campaigns, adjust pricing incentives, and personalize the user experience.

### Classification Target:
* **Target Column:** `Is_Returning_Customer`
* **Type:** Binary Boolean (`True` / `False`)
* **Business Value:** Identifying behavioral and demographic indicators that distinguish recurring high-value users from one-time shoppers.

---

## 2. Feature Schema & Leakage Mitigation

To build a robust predictive model capable of running at the moment of checkout (or session initiation), features must be explicitly filtered to prevent data leakage. Post-purchase metrics or columns that directly encode target attributes must be handled carefully.

### Feature Selection Matrix:
| Feature Category | Column Name | Action | Reason / Strategy |
|---|---|---|---|
| **Identifiers** | `Order_ID`, `Customer_ID` | **Drop** | Unique hash strings with no generalizable predictive power. |
| **Temporal** | `Date` | **Extract & Drop** | Transform into cyclical numerical inputs (`Month`, `DayOfWeek`). |
| **Demographics** | `Age`, `Gender`, `City` | **Keep** | Primary demographic pillars to identify regional and age-based retention trends. |
| **Session Behavior** | `Session_Duration_Minutes`, `Pages_Viewed`, `Device_Type` | **Keep** | Direct indicator of user engagement intensity during the purchase journey. |
| **Transaction Context** | `Product_Category`, `Unit_Price`, `Quantity`, `Payment_Method` | **Keep** | Cart composition and payment friction indicators. |
| **Financial Outcomes** | `Total sales`, `Discount sales`, `Discount_Percentage` | **Keep** | Financial value vectors associated with the current cart transaction. |
| **Post-Purchase** | `Delivery_Time_Days`, `Customer_Rating` | **Drop / Restrict** | These occur *after* the fulfillment cycle and cannot be used for real-time checkout prediction. |

---

## 3. Preprocessing & Engineering Architecture

The **antigravity** data engine will ingest raw inputs and process them through parallel transformation paths:
1. **Categorical Pipeline:** Multi-class strings (`City`, `Product_Category`, `Payment_Method`, `Gender`, `Device_Type`) are mapped into binary sparse matrices using One-Hot Encoding. Unknown strings encountered at inference are ignored gracefully (`handle_unknown='ignore'`).
2. **Numerical Pipeline:** Skewed continuous metrics (such as `Unit_Price` and `Total sales`) and integers are scaled using zero-mean variance normalization (`StandardScaler`) to aid model stability.
3. **Temporal Processing:** The `Date` timestamp is parsed to capture seasonal purchasing trends (e.g., month of the year, day of the week).

---

## 4. End-to-End Classification Pipeline Script

Below is a complete, structured Python implementation designed for the **antigravity** workspace to execute the **Customer Retention Classification** task.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix

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
    
    # 6. Model Definition (Gradient Boosting Classifier)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42))
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
    
    print("
================ CLASSIFICATION PERFORMANCE SUMMARY ================")
    print(f"Accuracy Score : {accuracy:.4f}")
    print(f"ROC-AUC Score  : {roc_auc:.4f}")
    print("
Confusion Matrix:")
    print(f"   Predicted Neg  Predicted Pos")
    print(f"Actual Neg: {cm[0][0]:<14} {cm[0][1]}")
    print(f"Actual Pos: {cm[1][0]:<14} {cm[1][1]}")
    
    print("
Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['New Customer (0)', 'Returning Customer (1)']))
    print("====================================================================
")
    
    return model_pipeline

if __name__ == "__main__":
    trained_classifier = run_antigravity_classification_pipeline("ecomerce_dataset.xlsx")
```

---

## 5. Deployment & Runtime Operations

To put this binary classifier into production within the **antigravity** ecosystem:
1. **Model Exporting:** Serialize the entire pipeline package using `joblib.dump(model_pipeline, 'antigravity_customer_retention_classifier.pkl')`. This bundles scaling factors, encoding maps, and the ensemble trees into a single artifact.
2. **Threshold Tuning:** The baseline implementation uses a classification decision threshold of `0.5`. For targeted marketing campaigns (e.g., sending high-value vouchers exclusively to churn-risk new customers), you can adjust the threshold on `predict_proba()` to optimize for higher **Precision** or **Recall** depending on business budgets.
