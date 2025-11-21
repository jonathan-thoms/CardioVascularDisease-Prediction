# 🫀 Multi-Source Cardiovascular Risk Prediction API

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/API-Flask-green)
![Sklearn](https://img.shields.io/badge/ML-Scikit--Learn-orange)

A robust Machine Learning API that predicts heart disease risk by synthesizing data from three major medical datasets: **Framingham Heart Study**, **MIMIC-III**, and **Kaggle Cardio Train**. 



## 🚀 Features

-   **Data Fusion:** Merges 3 heterogeneous datasets, standardizing medical coding (ICD-9) and units across sources.
-   **Robust Model:** Uses a `RandomForestClassifier` optimized via Grid Search, trained on a SMOTE-balanced dataset to handle class imbalance.
-   **Explainable AI:** Integrates **SHAP (SHapley Additive exPlanations)** to tell the user *why* they are at risk (e.g., "High Cholesterol is the top factor").
-   **Validation:** Strict input validation using `Pydantic` to prevent garbage data entry.

## 🏗️ Architecture

### The Data Pipeline
1.  **Ingestion:** Loads MIMIC-III (Clinical DB), Framingham (Longitudinal), and Cardio Train.
2.  **Preprocessing:** -   Aligns columns (e.g., mapping `sysBP` -> `trestbps`).
    -   Standardizes units (converting days to years).
    -   Imputes missing lifestyle factors (Smoking/Alcohol) where absent.
3.  **Training:** -   Data Split (80/20).
    -   **SMOTE** applied *only* to training data to prevent data leakage.
    -   Hyperparameter tuning via 5-Fold Cross-Validation.

## 🛠️ Installation

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/yourname/cardio-risk-api.git](https://github.com/yourname/cardio-risk-api.git)
    cd cardio-risk-api
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the model:**
    *(Requires dataset CSVs in `data/` folder)*
    ```bash
    python -m model.train_model
    ```

4.  **Run the API:**
    ```bash
    python app.py
    ```

## 🧪 Testing

Run the automated test suite to verify API health and validation logic:
```bash
python test_api.py
