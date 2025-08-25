from fastapi.testclient import TestClient
from model_service.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid():
    payload = {
        "gender": "Male",
        "age": 45,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 28.5,
        "HbA1c_level": 6.0,
        "blood_glucose_level": 140,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_label"] in {"low", "medium", "high"}


