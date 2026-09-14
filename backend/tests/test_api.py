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


def test_missing_activity_target():
    response = client.post(
        "/api/predict-activity",
        json={
            "smiles": "CCO",
            "target": "nonexistent"
        }
    )

    assert response.status_code == 404