import os
import joblib
import numpy as np
import shap
import traceback
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError, Field

# Initialize Flask
app = Flask(__name__)

# --- Data Validation Layer ---
class PatientData(BaseModel):
    age: float = Field(..., gt=0, lt=120)
    sex: int = Field(..., description="1 for Male, 0 for Female")
    trestbps: float = Field(..., gt=50, lt=250, description="Resting BP")
    diabp: float = Field(..., gt=30, lt=150, description="Diastolic BP")
    chol: float = Field(..., gt=100, lt=600)
    bmi: float = Field(..., gt=10, lt=60)
    glucose: float = Field(..., gt=50, lt=400)
    smoking: int = Field(..., ge=0, le=1)
    alcohol: int = Field(..., ge=0, le=1)
    exercise: int = Field(..., ge=0, le=1)

# --- Model Loading ---
MODEL_PATH = "model_artifacts/cardio_risk_model_v3.pkl"
model_data = None
model = None
features = None

def load_model():
    global model_data, model, features
    if os.path.exists(MODEL_PATH):
        model_data = joblib.load(MODEL_PATH)
        # Handle case where joblib loads just the model or the dict wrapper I created in train.py
        if isinstance(model_data, dict):
            model = model_data["model"]
            features = model_data["features"]
        else:
            model = model_data
            # Fallback features if loading old model format
            features = ["age", "gender", "trestbps", "diabp", "chol", "bmi", "glucose", "smoking", "alcohol", "exercise"]
        print("✅ Model loaded successfully.")
    else:
        print("⚠️ Model file not found. API will fail on predict.")

load_model()

# --- Helper Functions ---
def get_shap_explanation(input_array):
    """Calculates SHAP values safely."""
    try:
        # TreeExplainer is faster for Random Forest than generic Explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_array)
        
        # Handle binary classification output format (varies by sklearn version)
        if isinstance(shap_values, list):
            # Index 1 is usually the positive class (Risk)
            values = shap_values[1]
        else:
            # Some versions return shape (n_samples, n_features, n_classes)
            if len(shap_values.shape) == 3:
                values = shap_values[:, :, 1]
            else:
                values = shap_values

        # Flatten
        values = np.array(values).flatten()
        
        # Get top 3 drivers
        feature_impact = list(zip(features, values))
        # Sort by absolute impact magnitude
        feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return [f[0] for f in feature_impact[:3]]
    except Exception as e:
        print(f"SHAP Error: {e}")
        return ["Explanation unavailable"]

# --- Routes ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "online", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded on server"}), 503

    try:
        # 1. Validate Input using Pydantic
        json_data = request.get_json()
        patient = PatientData(**json_data)
        
        # 2. Prepare Data (Ensure order matches training)
        input_list = [
            patient.age, patient.sex, patient.trestbps, patient.diabp, 
            patient.chol, patient.bmi, patient.glucose, 
            patient.smoking, patient.alcohol, patient.exercise
        ]
        input_array = np.array([input_list], dtype=np.float32)

        # 3. Prediction
        probability = model.predict_proba(input_array)[0][1]
        risk_label = "High Risk" if probability > 0.5 else "Low Risk"

        # 4. Explanation (SHAP)
        important_factors = get_shap_explanation(input_array)

        return jsonify({
            "prediction": risk_label,
            "risk_probability": round(float(probability), 2),
            "key_risk_factors": important_factors
        })

    except ValidationError as e:
        return jsonify({"error": "Validation Failed", "details": e.errors()}), 400
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
