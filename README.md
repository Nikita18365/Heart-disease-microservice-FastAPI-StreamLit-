# Heart-disease-microservice-FastAPI-StreamLit-
Heart disease microservice (FastAPI + StreamLit)

- The project is designed to predict the likelihood of having a cardiovascular disease based on the patient's medical data.
- The model is trained on tabular medical signs and deployed via a FastAPI-based API with a Streamlit web interface.

### 📊 **Data**
Links to datasets:
- Train/Test: https://www.kaggle.com/competitions/playground-series-s6e2/data
- Origin: https://www.kaggle.com/datasets/cdeotte/s6e4-original-dataset

### 🧬 Features
The dataset contains both numerical and categorical medical features:

| Feature | Description |
|--------|-------------|
| **Age** | Patient age (years) |
| **Sex** | Gender (0 = Female, 1 = Male) |
| **Chest pain type** | Chest pain type (1–4) |
| **BP** | Resting blood pressure (mm Hg) |
| **Cholesterol** | Serum cholesterol (mg/dL) |
| **FBS over 120** | Fasting blood sugar > 120 mg/dL |
| **EKG results** | Resting ECG results (0–2) |
| **Max HR** | Maximum heart rate |
| **Exercise angina** | Exercise-induced angina |
| **ST depression** | ST depression induced by exercise |
| **Slope of ST** | Slope of the peak exercise ST segment |
| **Number of vessels fluro** | Number of major vessels by fluoroscopy |
| **Thallium** | Thallium stress test result |
| **Heart Disease** | Target variable (presence/absence of disease) |

### Project structure
project/

│

├── feature_engineering.py     # Feature generation and processing

├── app.py                     # FastAPI backend

├── streamlit_app.py           # Streamlit interface

│

├── model.joblib               # The trained model

├── artifacts.joblib           # Preprocessing, encoders, etc.

├── features.joblib            # List of used features

│

└── README.md

### Machine learning models
The project uses a simple trained XGBoost model for tabular data:
- trained on statistical encoded features
- optimized for ROC-AUC
- calibrated for probabilistic predictions

All necessary artifacts are preserved:
| Файл               | Назначение                           |
| ------------------ | ------------------------------------ |
| `model.joblib`     | the trained model                    |
| `artifacts.joblib` | preprocessing, encoder's, statistics |
| `features.joblib`  | list of model attributes             |

### Feature Engineering
1. Interaction Features
- Products and ratios of important features:
- age × cholesterol
- bp × age
- cholesterol / age
- st_slope_interaction
- exercise_st
and others.

2. Original Dataset Statistics
- Mean, count, std, skew and median statistics from the original dataset (`df_origin`) for numerical features.

3. Quantile Binning
- Continuous features were discretized using `pd.qcut`, producing `Quant_*` features.

4. Cross Features
- Pairwise combinations such as:
- `Quant_* X Chest pain type`
- other selected feature crosses.

5. Label & Frequency Encoding
- Label Encoding  
- Frequency Encoding  

### Base models (from Part I)
- **XGBoostClassifier (XGB)** — tree boosting using DMatrix (top 4 model)
- **LightGBMClassifier (LGB)** — gradient boosting with GPU/CPU support (top 3 model)
- **CatBoostClassifier (CAT)** — boosting with native pool container (top 2 model)
- **RealMLP (RealMLP_TD_Classifier)** - multi-layer Perceptron (top 1 model)

### Search for optimal hyperparameters (from Part I)
- Search for optimal hyperparameters of the model was carried out through the TPE algorithm of the Optuna package.

### Backend (FastAPI)
The API performs:
- loading of the model and artifacts
- application of feature engineering
- calculation of the probability of disease
- return the result in JSON

Request example:
- POST /predict

{
  "age": 54,
  "cholesterol": 230,
  "blood_pressure": 140
}

Answer:

{
  "probability": 0.82,
  "prediction": 1
}

### Frontend (Streamlit)
The file streamlit_app.py implements the user interface.

It allows you to:
- enter the patient's medical parameters
- send them to the API
- get the probability of the disease
- visually display the result

### Pipeline inference
User Input

      ↓
      
Streamlit UI

      ↓
      
FastAPI endpoint

      ↓
      
feature_engineering.py

      ↓
      
model.joblib

      ↓
      
The probability of disease
