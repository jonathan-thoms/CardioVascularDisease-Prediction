from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("cardio_risk_model.pkl")

# Define feature names (based on the dataset used)
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON request data
        data = request.get_json()

        # Ensure all required features are present
        if not all(feature in data for feature in FEATURES):
            return jsonify({"error": "Missing required features"}), 400

        # Convert input data to NumPy array for prediction
        input_data = np.array([data[feature] for feature in FEATURES]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Map prediction to meaningful output
        risk_level = "High Risk" if prediction == 1 else "Low Risk"

        # Return result
        return jsonify({
            "prediction": risk_level,
            "probability": round(probability, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
