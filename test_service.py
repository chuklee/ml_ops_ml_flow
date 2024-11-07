import requests
import numpy as np
from sklearn import datasets

def test_prediction_endpoint():
    # Load some test data
    iris = datasets.load_iris()
    X = iris.data[:5].tolist()  # Convert to list for JSON serialization
    
    # Test prediction endpoint
    response = requests.post(
        "http://localhost:8000/predict",
        json={"features": X}
    )
    assert response.status_code == 200
    print("Prediction test passed!")
    
def test_model_update():
    # Test model update endpoint
    response = requests.post(
        "http://localhost:8000/update-model",
        json={"version": "1"}
    )
    assert response.status_code == 200
    
    # Test accept-next-model endpoint
    response = requests.post("http://localhost:8000/accept-next-model")
    assert response.status_code == 200
    print("Model update test passed!")

if __name__ == "__main__":
    test_prediction_endpoint()
    test_model_update() 