Heart-disease-predictor

# 🫀 Heart Disease Prediction using Medallion Architecture + FastAPI

This project implements an end-to-end **machine learning pipeline** using the Heart Disease UCI dataset with a **Medallion architecture** in MongoDB. Predictions are served via a **FastAPI REST API**, deployed on **Render**, and interactively tested through **Swagger UI**.

---

## 📌 Project Overview

- Dataset: [Kaggle - Heart Disease UCI](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- Architecture: Medallion (Bronze → Silver → Gold) using MongoDB Atlas
- Models: Logistic Regression, Random Forest, XGBoost
- API: FastAPI with Swagger documentation
- Hosting: Render
- Output: `.pkl` model file, interactive API, GitHub repository, and screenshots

---

## 🗂️ Medallion Architecture (MongoDB)

### 🟫 Bronze Layer
- Raw CSV inserted as-is without preprocessing
- Stored as JSON documents
- **Collection**: `heart_disease_bronze`  
- **Database**: `healthcare`

### 🟪 Silver Layer
- Preprocessing includes:
  - Missing value imputation (numerical → mean, categorical → mode)
  - Label encoding of categorical columns (`sex`, `cp`, `thal`, etc.)
- Stored in `heart_disease_silver`

### 🟨 Gold Layer
- Final refined data:
  - MinMax normalization on numeric features
  - Feature selection based on correlation and model importance
- Stored in `heart_disease_gold`

📸 **Screenshots stored in `screenshots/`**:
- `bronze_sample.png`
- `silver_sample.png`
- `gold_sample.png`

---

## ⚙️ Model Development

📁 Notebook: `model_training.ipynb`

### Models Evaluated
- Logistic Regression
- Random Forest ✅ *(Best Performing)*
- XGBoost

### Why Random Forest?
- Balanced accuracy/precision/recall
- Feature importance clearly explainable
- Robust to outliers and non-linear patterns

### Preprocessing Highlights
- Missing value imputation
- Label encoding
- MinMax scaling
- Feature selection using correlation heatmap + Random Forest importance

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC (Random Forest: **0.89+**)

### Final Step
Model saved as `.pkl` using joblib:
```python
joblib.dump(best_model, "heart_disease_model.pkl")
