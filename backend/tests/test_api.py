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

def test_model_manifest_is_explicit():
    v=client.get("/api/version").json()
    assert v["application_version"]=="4.0.0"
    assert v["model_version"]["model_bundle_version"]=="3.1.0"
    assert v["model_version"]["application_version"]=="4.0.0"

def test_target_catalog_matches_bundled_availability():
    r=client.get("/api/activity-targets")
    assert r.status_code==200
    for target in r.json()["targets"]:
        assert target["name"]
        assert target["target_chembl_id"]

def test_local_sensitivity_uses_real_perturbations():
    d=client.post("/api/analyze",json={"smiles":"CCO"}).json()
    rows=d["local_sensitivity"]["features"]
    assert len(rows)==12
    assert all("solubility_delta_plus_1sd" in row for row in rows)
    assert all("toxicity_probability_delta_plus_1sd" in row for row in rows)
    assert any(abs(row["solubility_delta_plus_1sd"])>0 or abs(row["toxicity_probability_delta_plus_1sd"])>0 for row in rows)

def test_chemical_space_single_molecule_api():
    r=client.post("/api/chemical-space",json={"smiles":["CCO"]})
    assert r.status_code==200
    d=r.json()
    assert d["clusters"]==1
    assert len(d["points"])==1

def test_async_screen_job_lifecycle():
    r=client.post("/api/screen/jobs",files={"file":("x.smi",b"CCO\nCCCO\n","text/plain")})
    assert r.status_code==200
    job=r.json()
    assert job["job_id"].startswith("screen_")
    status=client.get("/api/screen/jobs/"+job["job_id"])
    assert status.status_code==200

def test_async_screen_rejects_oversized_batch():
    from config import MAX_BATCH_MOLECULES
    payload=("CCO\n"* (MAX_BATCH_MOLECULES+1)).encode()
    r=client.post("/api/screen/jobs",files={"file":("large.smi",payload,"text/plain")})
    assert r.status_code==413


def test_security_headers():
    r=client.get("/api/health")
    assert r.headers["x-content-type-options"]=="nosniff"
    assert r.headers["x-frame-options"]=="DENY"
    assert "x-request-id" in r.headers
    assert "x-process-time-ms" in r.headers

def test_async_screen_rejects_unsupported_file_type():
    r=client.post("/api/screen/jobs",files={"file":("input.exe",b"CCO\n","application/octet-stream")})
    assert r.status_code==400
