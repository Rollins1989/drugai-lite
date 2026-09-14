from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_activity_targets():
    response = client.get("/api/activity-targets")
    assert response.status_code == 200

    data = response.json()
    assert "targets" in data


def test_activity_prediction():
    response = client.post(
        "/api/predict-activity",
        json={
            "smiles": "CCO",
            "target": "egfr"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "rf_pIC50" in data
    assert "gb_pIC50" in data
    assert "confidence" in data