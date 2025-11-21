import pandas as pd
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# Load Framingham Heart Study dataset
framingham_data = pd.read_csv("data/framingham.csv")

# Load MIMIC-III datasets
patients = pd.read_csv("data/patients.csv", usecols=["subject_id", "gender", "dob"])
admissions = pd.read_csv("data/admissions.csv", usecols=["subject_id", "admittime", "diagnosis"])
labevents = pd.read_csv("data/labevents.csv", usecols=["subject_id", "itemid", "valuenum"])
chartevents = pd.read_csv("data/chartevents.csv", usecols=["subject_id", "itemid", "valuenum"])
diagnoses = pd.read_csv("data/diagnoses_icd.csv", usecols=["subject_id", "icd9_code"])

# Convert gender to numeric
patients["gender"] = patients["gender"].map({"M": 1, "F": 0})

# Process lab and chart events (selecting relevant measurements)
lab_items = {50907: "chol", 50882: "glucose"}
chart_items = {220045: "trestbps", 220210: "diabp", 220277: "bmi"}

labevents = labevents[labevents["itemid"].isin(lab_items.keys())].replace({"itemid": lab_items})
chartevents = chartevents[chartevents["itemid"].isin(chart_items.keys())].replace({"itemid": chart_items})

# Handle duplicate values by averaging per subject_id
labevents = labevents.groupby(["subject_id", "itemid"])["valuenum"].mean().unstack().reset_index()
chartevents = chartevents.groupby(["subject_id", "itemid"])["valuenum"].mean().unstack().reset_index()

# Merge MIMIC datasets
mimic_data = patients.merge(admissions, on="subject_id", how="inner")
mimic_data = mimic_data.merge(labevents, on="subject_id", how="left")
mimic_data = mimic_data.merge(chartevents, on="subject_id", how="left")
mimic_data = mimic_data.merge(diagnoses, on="subject_id", how="left")

# Convert to datetime, handling future dates by shifting 100 years
mimic_data["dob"] = pd.to_datetime(mimic_data["dob"], errors="coerce")
mimic_data["admittime"] = pd.to_datetime(mimic_data["admittime"], errors="coerce")

current_year = datetime.datetime.now().year
mimic_data.loc[mimic_data["dob"].dt.year > current_year, "dob"] -= pd.DateOffset(years=100)
mimic_data.loc[mimic_data["admittime"].dt.year > current_year, "admittime"] -= pd.DateOffset(years=100)

# Compute age at admission
mimic_data["age"] = mimic_data["admittime"].dt.year - mimic_data["dob"].dt.year
mimic_data.loc[mimic_data["age"] > 110, "age"] = 90
mimic_data.loc[mimic_data["age"] < 0, "age"] = np.nan
mimic_data.dropna(subset=["age"], inplace=True)

# Drop unnecessary columns
mimic_data.drop(columns=["subject_id", "admittime", "dob"], inplace=True)

# Define target variable (heart disease risk based on ICD9 codes)
mimic_data["target"] = mimic_data["icd9_code"].apply(lambda x: 1 if str(x).startswith("410") else 0)
mimic_data.drop(columns=["icd9_code"], inplace=True)

# Load Cardiovascular Disease Dataset
cardio_data = pd.read_csv("data/cardio_train.csv", sep=";")

# Check if BMI is missing
if "bmi" not in cardio_data.columns:
    # Calculate BMI using weight (kg) and height (cm)
    cardio_data["bmi"] = cardio_data["weight"] / ((cardio_data["height"] / 100) ** 2)

# Rename columns to match existing datasets
cardio_data.rename(columns={
    "age": "age",
    "gender": "gender",
    "ap_hi": "trestbps",
    "ap_lo": "diabp",
    "cholesterol": "chol",
    "gluc": "glucose",
    "smoke": "smoking",
    "alco": "alcohol",
    "active": "exercise",
    "cardio": "target"
}, inplace=True)

# Convert age from days to years
cardio_data["age"] = cardio_data["age"] // 365

# Normalize categorical columns (0/1 encoding)
cardio_data["gender"] = cardio_data["gender"].map({1: 1, 2: 0})
cardio_data["smoking"] = cardio_data["smoking"].astype(int)
cardio_data["alcohol"] = cardio_data["alcohol"].astype(int)
cardio_data["exercise"] = cardio_data["exercise"].astype(int)

# Define feature set
common_features = ["age", "gender", "trestbps", "diabp", "chol", "bmi", "glucose", "smoking", "alcohol", "exercise", "target"]

# Standardize column names for Framingham dataset
framingham_data.rename(columns={
    "male": "gender",
    "sysBP": "trestbps",
    "diaBP": "diabp",
    "totChol": "chol",
    "BMI": "bmi",
    "glucose": "glucose",
    "TenYearCHD": "target"
}, inplace=True)

# Ensure missing columns exist in all datasets
for col in ["smoking", "alcohol", "exercise"]:
    if col not in framingham_data:
        framingham_data[col] = 0  # Default: Non-smoker, No alcohol, No exercise

    if col not in mimic_data:
        mimic_data[col] = 0  # Default: Non-smoker, No alcohol, No exercise

# Select only relevant columns
framingham_data = framingham_data[common_features[:-3]]  # Framingham lacks smoking, alcohol, exercise
mimic_data = mimic_data[common_features[:-3]]  # MIMIC lacks lifestyle factors
cardio_data = cardio_data[common_features]  # Cardio dataset has all features

# Merge all datasets
data = pd.concat([framingham_data, mimic_data, cardio_data], ignore_index=True)
data.dropna(inplace=True)

# Define features and target
X = data.drop(columns=["target"])
y = data["target"]

# Apply SMOTE to balance dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data **AFTER SMOTE**
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}

# Perform Grid Search with 5-fold Cross Validation
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Train the best model
best_model.fit(X_train, y_train)

# Evaluate model
y_pred = best_model.predict(X_test)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save best model
joblib.dump(best_model, "cardio_risk_model_v3.pkl")
print("Optimized Model saved as cardio_risk_model_v3.pkl")
