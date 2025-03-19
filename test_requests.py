import requests

# API URL
url = "http://127.0.0.1:5000/predict"

# JSON data (patient details)
data = {
    "age": 68,
    "sex": 1,
    "cp": 1,
    "trestbps": 160,
    "chol": 290,
    "fbs": 1,
    "restecg": 2,
    "thalach": 120,
    "exang": 1,
    "oldpeak": 2.5,
    "slope": 1,
    "ca": 3,
    "thal": 3
}

# Send POST request
response = requests.post(url, json=data)

# Print the response
print(response.json())
