import traceback  # Add this to capture detailed errors
from flask import Flask, request, jsonify
import joblib
import numpy as np
import shap
import os
# Load the trained model
model = joblib.load("cardio_risk_model_v3.pkl")

# Define feature names
FEATURES = ["age", "sex", "trestbps", "diabp", "chol", "bmi", "glucose", "smoking", "alcohol", "exercise"]

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON request data
        data = request.get_json()

        # Check if all required features are present
        missing_features = [feature for feature in FEATURES if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing required features: {missing_features}"}), 400

        # Convert input data to NumPy array
        input_data = np.array([[data[feature] for feature in FEATURES]], dtype=np.float32)  # Ensure correct shape

        # Make prediction
        probability = model.predict_proba(input_data)[0][1]  # Probability of heart disease
        prediction = "High Risk" if probability > 0.5 else "Low Risk"

        # SHAP Explainability
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(input_data)

        # If the model is binary, get SHAP values for the positive class (1)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get SHAP values for class 1 (Heart Disease)

        # Ensure `shap_values` is converted to a NumPy array
        shap_values = np.array(shap_values)

        # Flatten the array if needed
        if shap_values.ndim > 1:
            shap_values = shap_values.flatten()

        # Extract top 3 important features
        important_factors = sorted(zip(FEATURES, shap_values), key=lambda x: abs(x[1]), reverse=True)[:3]
        important_factors = [factor[0] for factor in important_factors]

        # Return result
        return jsonify({
            "prediction": prediction,
            "probability": round(probability, 2),
            "important_factors": important_factors
        })

    except Exception as e:
        error_details = traceback.format_exc()
        print(error_details)  # Print full error in terminal
        return jsonify({"error": str(e), "details": error_details}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 10000 if PORT is not set
    app.run(host='127.0.0.1', port=port)

