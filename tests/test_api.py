from fastapi.testclient import TestClient
from toronto_transit_ml_platform.api import app

client = TestClient(app)

def test_health():
    output = client.get("/health")
    
    assert output.status_code == 200
    assert output.json() == {"status": "healthy"}

def test_predict_valid(monkeypatch):
    monkeypatch.setattr(
	"toronto_transit_ml_platform.api.save_prediction",
	lambda *args, **kwargs: None,
	)
    response = client.post(
        "/predict",
        json={
              "day": "Wednesday",
              "line": "102 MARKHAM ROAD",
              "code": "MFDV",
              "bound": "N",
              "month": 8,
              "hour": 17,
              },
        )

    assert response.status_code == 200
    assert "predicted_delay_minutes" in response.json()

def test_predict_invalid_month():
    response = client.post(
        "/predict",
        json={
            "day": "Wednesday",
            "line": "102 MARKHAM ROAD",
            "code": "MFDV",
            "bound": "N",
            "month": 15,
            "hour": 17,
            },
    )
    
    assert response.status_code == 422

def test_metrics():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "prediction_requests_total" in response.text
    assert "prediction_latency_seconds" in response.text
