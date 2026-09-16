from main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "DrugAI" in response.text


def test_health_contract():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "reference_library_size" in data
    assert "targets" in data
    assert "model_version" in data


def test_version_contract():
    response = client.get("/api/version")
    assert response.status_code == 200
    data = response.json()
    assert "application_version" in data
    assert "model_version" in data


def test_model_cards_document_uncertainty_score_and_metrics():
    response = client.get("/api/model-cards")
    assert response.status_code == 200
    data = response.json()
    assert "interpretation" in data
    assert "model_agreement" in data["interpretation"]
    assert "screening_score" in data["interpretation"]
    assert "limitations" in data
    assert "solubility" in data["metrics"]
    assert "toxicity" in data["metrics"]
    assert "model_version" in data


def test_analyze_returns_scientifically_named_outputs():
    response = client.post("/api/analyze", json={"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert "screening_score" in data
    assert 0 <= data["screening_score"]["value"] <= 1
    assert data["screening_score"]["type"] == "heuristic"
    assert abs(sum(data["screening_score"]["weights"].values()) - 1.0) < 1e-9
    assert "model_agreement" in data["solubility"]
    assert "model_agreement" in data["toxicity"]
    assert "confidence" not in data["solubility"]
    assert "confidence" not in data["toxicity"]
    assert "method" in data["applicability_domain"]


def test_invalid_smiles():
    response = client.post("/api/analyze", json={"smiles": "not-a-valid-smiles"})
    assert response.status_code == 400


def test_request_validation_rejects_empty_smiles():
    response = client.post("/api/analyze", json={"smiles": ""})
    assert response.status_code == 422


def test_screen_rejects_unsupported_extension():
    response = client.post("/api/screen", files={"file": ("library.pdf", b"CCO\n", "application/pdf")})
    assert response.status_code == 400


def test_screen_accepts_small_smi_batch():
    response = client.post("/api/screen", files={"file": ("library.smi", b"CCO\nCCC\n", "text/plain")})
    assert response.status_code == 200
    data = response.json()
    assert data["funnel"]["submitted"] == 2
    assert "candidates" in data


def test_activity_targets():
    response = client.get("/api/activity-targets")
    assert response.status_code == 200
    assert "targets" in response.json()


def test_missing_activity_target():
    response = client.post("/api/predict-activity", json={"smiles": "CCO", "target": "nonexistent"})
    assert response.status_code == 404
