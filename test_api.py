import unittest
import json
from app import app

class TestCardioAPI(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_check(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)

    def test_prediction_valid_data(self):
        payload = {
            "age": 55,
            "sex": 1,
            "trestbps": 140,
            "diabp": 90,
            "chol": 240,
            "bmi": 28.5,
            "glucose": 100,
            "smoking": 0,
            "alcohol": 1,
            "exercise": 0
        }
        response = self.app.post('/predict', 
                                 data=json.dumps(payload), 
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("prediction", data)
        self.assertIn("risk_probability", data)
        self.assertIsInstance(data["key_risk_factors"], list)

    def test_validation_error(self):
        # Sending "age": "old" should trigger error
        payload = {
            "age": "old", 
            "sex": 1,
            "trestbps": 140, "diabp": 90, "chol": 240,
            "bmi": 28.5, "glucose": 100, "smoking": 0,
            "alcohol": 1, "exercise": 0
        }
        response = self.app.post('/predict', 
                                 data=json.dumps(payload), 
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("Validation Failed", response.get_json()["error"])

if __name__ == "__main__":
    unittest.main()
