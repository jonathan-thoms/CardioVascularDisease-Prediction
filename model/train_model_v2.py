import pandas as pd
import numpy as np
import datetime
import joblib
import os
from typing import Tuple, List
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# Ensure directories exist
os.makedirs("model_artifacts", exist_ok=True)

def process_mimic(patients_path: str, admissions_path: str, lab_path: str, chart_path: str, diag_path: str) -> pd.DataFrame:
    """
    Preprocesses MIMIC-III data: merges tables, handles dates, and calculates age.
    """
    print("Loading MIMIC-III dataset...")
    try:
        # Load raw files (Optimized with specific columns)
        patients = pd.read_csv(patients_path, usecols=["subject_id", "gender", "dob"])
        admissions = pd.read_csv(admissions_path, usecols=["subject_id", "admittime"])
        labevents = pd.read_csv(lab_path, usecols=["subject_id", "itemid", "valuenum"])
        chartevents = pd.read_csv(chart_path, usecols=["subject_id", "itemid", "valuenum"])
        diagnoses = pd.read_csv(diag_path, usecols=["subject_id", "icd9_code"])

        # Standardization
        patients["gender"] = patients["gender"].map({"M": 1, "F": 0})

        # Feature Mapping
        lab_map = {50907: "chol", 50882: "glucose"}
        chart_map = {220045: "trestbps", 220210: "diabp", 220277: "bmi"}

        # Filter and Rename
        labevents = labevents[labevents["itemid"].isin(lab_map.keys())].replace({"itemid": lab_map})
        chartevents = chartevents[chartevents["itemid"].isin(chart_map.keys())].replace({"itemid": chart_map})

        # Aggregate (Mean per patient)
        lab_agg = labevents.groupby(["subject_id", "itemid"])["valuenum"].mean().unstack().reset_index()
        chart_agg = chartevents.groupby(["subject_id", "itemid"])["valuenum"].mean().unstack().reset_index()

        # Merging
        df = patients.merge(admissions, on="subject_id", how="inner")
        df = df.merge(lab_agg, on="subject_id", how="left")
        df = df.merge(chart_agg, on="subject_id", how="left")
        df = df.merge(diagnoses, on="subject_id", how="left")

        # Date Handling (Shift future dates)
        for col in ["dob", "admittime"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            current_year = datetime.datetime.now().year
            df.loc[df[col].dt.year > current_year, col] -= pd.DateOffset(years=100)

        # Age Calculation
        df["age"] = df["admittime"].dt.year - df["dob"].dt.year
        df.loc[(df["age"] > 110) | (df["age"] < 0), "age"] = np.nan
        df.dropna(subset=["age"], inplace=True)

        # Target Generation (ICD9 starts with 410 = Heart Attack)
        df["target"] = df["icd9_code"].apply(lambda x: 1 if str(x).startswith("410") else 0)

        # Add missing columns common to other datasets
        for col in ["smoking", "alcohol", "exercise"]:
            df[col] = 0 

        return df[["age", "gender", "trestbps", "diabp", "chol", "bmi", "glucose", "smoking", "alcohol", "exercise", "target"]]
    
    except FileNotFoundError as e:
        print(f"Skipping MIMIC: {e}")
        return pd.DataFrame()

def process_cardio(path: str) -> pd.DataFrame:
    print("Loading Cardio Train dataset...")
    try:
        df = pd.read_csv(path, sep=";")
        
        # BMI Calculation
        if "bmi" not in df.columns:
            df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

        # Renaming
        rename_map = {
            "age": "age", "gender": "gender", "ap_hi": "trestbps", 
            "ap_lo": "diabp", "cholesterol": "chol", "gluc": "glucose", 
            "smoke": "smoking", "alco": "alcohol", "active": "exercise", 
            "cardio": "target"
        }
        df.rename(columns=rename_map, inplace=True)

        # Normalization
        df["age"] = df["age"] // 365
        df["gender"] = df["gender"].map({1: 1, 2: 0}) # Assuming 1=Male
        
        return df[list(rename_map.values())]
    except FileNotFoundError:
        print("Skipping Cardio Train (File not found)")
        return pd.DataFrame()

def process_framingham(path: str) -> pd.DataFrame:
    print("Loading Framingham dataset...")
    try:
        df = pd.read_csv(path)
        rename_map = {
            "male": "gender", "sysBP": "trestbps", "diaBP": "diabp",
            "totChol": "chol", "BMI": "bmi", "glucose": "glucose", 
            "TenYearCHD": "target"
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Add missing cols
        for col in ["smoking", "alcohol", "exercise"]:
            df[col] = 0

        cols = list(rename_map.values()) + ["smoking", "alcohol", "exercise"]
        return df[cols]
    except FileNotFoundError:
        print("Skipping Framingham (File not found)")
        return pd.DataFrame()

def train():
    # 1. Load and Merge Data
    df_mimic = process_mimic("data/patients.csv", "data/admissions.csv", "data/labevents.csv", "data/chartevents.csv", "data/diagnoses_icd.csv")
    df_cardio = process_cardio("data/cardio_train.csv")
    df_fram = process_framingham("data/framingham.csv")

    combined_df = pd.concat([df_mimic, df_cardio, df_fram], ignore_index=True)
    combined_df.dropna(inplace=True)

    print(f"Total records after merging: {len(combined_df)}")

    X = combined_df.drop(columns=["target"])
    y = combined_df["target"]

    # 2. Split Data BEFORE SMOTE (Crucial for validity)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Apply SMOTE only to Training Data
    print("Applying SMOTE to training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 4. Hyperparameter Tuning
    print("Tuning Hyperparameters...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }
    
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train_res, y_train_res)

    best_model = grid.best_estimator_

    # 5. Evaluation
    y_pred = best_model.predict(X_test)
    print(f"Best Params: {grid.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred):.4f}")

    # 6. Save Model & Feature Names
    # Saving columns is important to ensure API inputs match Model inputs order
    model_data = {
        "model": best_model,
        "features": list(X.columns)
    }
    joblib.dump(model_data, "model_artifacts/cardio_risk_model_v3.pkl")
    print("Model saved successfully.")

if __name__ == "__main__":
    train()
