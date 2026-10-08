from chemistry import compute_descriptors, parse_mol
from services import parse_uploaded_records
from model_service import screening_score

def test_descriptors_deterministic():
    m=parse_mol("CCO")
    assert compute_descriptors(m)==compute_descriptors(m)

def test_upload_csv_preserves_metadata():
    rows=parse_uploaded_records("compound_id,smiles\nA,CCO\nB,CCCO\n")
    assert rows[0]["compound_id"]=="A"
    assert rows[1]["smiles"]=="CCCO"

def test_score_weights():
    r=screening_score({"label":"GOOD"},{"toxicity_probability":.2},.8,True,True,True,"HIGH")
    assert abs(sum(r["weights"].values())-1)<1e-9
