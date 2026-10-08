from main import app
from fastapi.testclient import TestClient

client=TestClient(app)

def test_health_and_version():
    assert client.get("/api/health").status_code==200
    v=client.get("/api/version").json()
    assert v["application_version"]=="4.0.0"
    assert "python_version" in v

def test_analyze_contract():
    r=client.post("/api/analyze",json={"smiles":"CCO"})
    assert r.status_code==200
    d=r.json()
    assert d["valid"] is True
    assert "local_sensitivity" in d
    assert "empirical_ensemble_interval_80pct" in d["solubility"]
    assert 0<=d["screening_score"]["value"]<=1

def test_invalid_smiles():
    assert client.post("/api/analyze",json={"smiles":"not-valid"}).status_code==400

def test_compare():
    r=client.post("/api/compare",json={"smiles_a":"CCO","smiles_b":"CCCO"})
    assert r.status_code==200
    assert 0<=r.json()["structural_similarity"]["tanimoto"]<=1

def test_chemical_space():
    r=client.post("/api/chemical-space",json={"smiles":["CCO","CCCO","c1ccccc1"]})
    assert r.status_code==200
    assert len(r.json()["points"])==3

def test_pareto():
    rows=[{"activity":8,"toxicity_probability":.1,"qed":.8},{"activity":7,"toxicity_probability":.3,"qed":.7}]
    r=client.post("/api/pareto",json={"rows":rows,"objectives":["activity","toxicity","qed"]})
    assert r.status_code==200
    assert any(x["pareto_optimal"] for x in r.json()["rows"])

def test_model_cards():
    d=client.get("/api/model-cards").json()
    assert "interpretation" in d and "limitations" in d
    assert "uncertainty" in d["interpretation"]

def test_screen_small_library():
    r=client.post("/api/screen",files={"file":("x.smi",b"CCO\nCCCO\n","text/plain")})
    assert r.status_code==200
    d=r.json()
    assert d["candidate_count"]>=1
    assert "all_candidates" in d
