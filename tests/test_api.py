import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api.main import app, predictor, advisor

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_dependencies():
    # Mock the predictor model so it doesn't need actual trained models to run tests
    class MockModel:
        def predict(self, features):
            return [15.5]
    
    predictor.model = MockModel()
    predictor.features = ['sensor_1', 'sensor_1_roll_mean_5']
    
    # We mock Gemini client directly in the test where needed
    yield

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "AeroPredict AI" in response.json()["message"]

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid_payload():
    payload = {
        "engine_id": 1,
        "history": [
            {"time_cycles": 1, "sensor_1": 10.0},
            {"time_cycles": 2, "sensor_1": 12.0}
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_rul"] == 15.5
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

def test_predict_invalid_payload():
    payload = {
        "engine_id": "not_an_int",
        "history": "not_a_list"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity

def test_explain_valid_payload():
    payload = {
        "engine_id": 1,
        "history": [
            {"time_cycles": 1, "sensor_1": 10.0},
            {"time_cycles": 2, "sensor_1": 12.0}
        ]
    }
    # Mock the explain_prediction function so we don't need SHAP to actually run
    with patch("api.main.explain_prediction") as mock_explain:
        mock_explain.return_value = [{"name": "sensor_1", "impact": 0.5}]
        response = client.post("/explain", json=payload)
        
        assert response.status_code == 200
        assert len(response.json()["top_features"]) == 1

def test_recommendation_with_mocked_gemini():
    payload = {
        "engine_id": 1,
        "history": [
            {"time_cycles": 1, "sensor_1": 10.0},
            {"time_cycles": 2, "sensor_1": 12.0}
        ]
    }
    
    with patch("api.main.explain_prediction") as mock_explain:
        mock_explain.return_value = [{"name": "sensor_1", "impact": 0.5}]
        
        with patch.object(advisor, "get_recommendation", return_value="Mocked advice.") as mock_get_rec:
            response = client.post("/recommendation", json=payload)
            
            assert response.status_code == 200
            assert response.json()["recommendation"] == "Mocked advice."
            mock_get_rec.assert_called_once()
