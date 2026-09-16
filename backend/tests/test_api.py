from main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health_contract():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "reference_library_size" in data
    assert "targets" in data


def test_model_cards_document_uncertainty_and_score():
    response = client.get("/api/model-cards")
    assert response.status_code == 200
    data = response.json()
    assert "interpretation" in data
    assert "model_agreement" in data["interpretation"]
    assert "screening_score" in data["interpretation"]
    assert "limitations" in data


def test_analyze_returns_scientifically_named_outputs():
    response = client.post(
        "/api/analyze",
        json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert "screening_score" in data
    assert "value" in data["screening_score"]
    assert data["screening_score"]["type"] == "heuristic"
    assert "model_agreement" in data["solubility"]
    assert "model_agreement" in data["toxicity"]
    assert "confidence" not in data["solubility"]
    assert "confidence" not in data["toxicity"]
    assert "method" in data["applicability_domain"]


def test_invalid_smiles():
    response = client.post(
        "/api/analyze",
        json={"smiles": "not-a-valid-smiles"},
    )
    assert response.status_code == 400


def test_activity_targets():
    response = client.get("/api/activity-targets")
    assert response.status_code == 200
    data = response.json()
    assert "targets" in data


def test_missing_activity_target():
    response = client.post(
        "/api/predict-activity",
        json={"smiles": "CCO", "target": "nonexistent"},
    )
    assert response.status_code == 404
