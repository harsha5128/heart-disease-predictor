#   Heart Disease Prediction using Medallion Architecture + FastAPI

This project implements a complete **Machine Learning pipeline** for heart disease prediction using the **Medallion Architecture** in MongoDB. It includes **Data Preprocessing**, **Model Training**, and a **FastAPI REST API**, all deployed on **Render** and tested through **Swagger UI** and **Postman**.

---

##               Project Overview

- **Dataset**: [Kaggle - Heart Disease UCI](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- **Architecture**: Medallion (Bronze → Silver → Gold) using MongoDB Atlas
- **Models Evaluated**: Logistic Regression, Random Forest, XGBoost
- **Final Model**: ✅ **XGBoost** based on all metrics 
- **Deployment**: fastAPI on Render
- **Testing Tools**: Swagger UI & Postman

---

##       Medallion Architecture (MongoDB)

###  Bronze Layer
- Raw heart disease CSV ingested directly into MongoDB.
- No transformations applied.
- **Collection**: `heart_disease_bronze`
- **Purpose**: Immutable, raw input archive.

###  Silver Layer
- Preprocessing steps:
  - Missing value imputation (mean/mode)
  - Label encoding (`sex`, `cp`, `thal`, etc.)
- **Collection**: `heart_disease_silver`
- **Purpose**: Clean, structured, ML-ready data.

###  Gold Layer
- Final transformation steps:
  - MinMax Scaling
  - Feature selection based on correlation and model importance
- Used for training final model.
- **Collection**: `heart_disease_gold`
- **Shape**: `(920, 10)`

    **MongoDB Screenshots**:
- `screenshots/mongohealthbronze.png`
- `screenshots/mongohealthsilver.png`
- `screenshots/mongohealthgold.png`

---

##  Model Development and Evaluation

 Notebook: `notebooks/model_training.ipynb`

###  Steps Followed
1. Exploratory Data Analysis (EDA)
2. Handling missing values
3. Label encoding for categorical variables
4. Feature scaling (MinMax)
5. Model training using multiple classifiers
6. Evaluation using various metrics
7. Saving best model as `.pkl` using `joblib`

```python
joblib.dump(best_model, "heart_disease_model.pkl")
```

###  Models Compared

| Model              | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.84     | 0.83      | 0.82   | 0.82     | 0.86    |
| Random Forest       | 0.87     | 0.86      | 0.87   | 0.86     | 0.89    |
| **XGBoost** ✅       | **0.89** | **0.89**  | **0.88** | **0.89** | **0.91** |

###  Why XGBoost?
- Best ROC-AUC (**0.91**), indicating top-ranking ability.
- Excellent balance in precision, recall, and F1.
- Handles outliers, feature interactions, and scale robustly.
- Most generalizable model for clinical risk detection.

---

##  FastAPI + MongoDB API

###  Render Deployment (Live API)
 [https://heart-disease-predictor-ylvv.onrender.com](#) 

###  API Endpoints

| Method | Endpoint     | Description                  |
|--------|--------------|------------------------------|
| GET    | `/`          | Welcome message              |
| GET    | `/health`    | Health check endpoint        |
| POST   | `/predict`   | Predict heart disease        |

---

##  Postman Instructions

1. Open Postman.
2. Method: `POST`
3. URL: `http://localhost:8000/predict`
4. Body: `raw` → `JSON`

```json
{
  "oldpeak": 1.5,
  "thalch": 150,
  "exang": 0,
  "age": 54,
  "ca": 0,
  "cp": 1,
  "dataset": 1,
  "id": 1001,
  "sex": 1
}
```

 Screenshots available in `screenshots/postrequest.png`

---

##  Setup Instructions

###  Environment Setup

```bash
git clone https://github.com/harsha5128/heart-disease-predictor.git
cd heart-disease-predictor
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

###  Run MongoDB (if local)
- Ensure MongoDB is running locally or use MongoDB Atlas.

###  Launch API

```bash
cd api
uvicorn main:app --reload  or uvicorn api.main:app --reload
```

- Open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---



---

##  Submission Checklist

| Criteria               | ✅ Done |
|------------------------|--------|
| Medallion Architecture | ✅      |
| Model Training         | ✅      |
| API Development        | ✅      |
| Render Deployment      | ✅      |
| MongoDB Screenshots    | ✅      |
| Postman Testing        | ✅      |
| Final Model Justified  | ✅      |
| README Documentation   | ✅      |