from chemistry import compute_descriptors, parse_mol
from services import domain_signal, parse_uploaded_smiles
from model_service import screening_score


def test_parse_mol_and_descriptors_are_deterministic():
    mol = parse_mol("CCO")
    first = compute_descriptors(mol)
    second = compute_descriptors(mol)
    assert first == second
    assert first["MolWt"] > 0
    assert 0 <= first["QED"] <= 1


def test_domain_signal_thresholds():
    assert domain_signal([])["status"] == "UNKNOWN"
    assert domain_signal([{"tanimoto": 0.75}])["status"] == "HIGH"
    assert domain_signal([{"tanimoto": 0.50}])["status"] == "MODERATE"
    assert domain_signal([{"tanimoto": 0.20}])["status"] == "LOW"


def test_screening_score_is_bounded_and_weights_sum_to_one():
    sol = {"label": "GOOD"}
    tox = {"toxicity_probability": 0.2}
    result = screening_score(sol, tox, 0.7, True, True, True, "HIGH")
    assert 0 <= result["value"] <= 1
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-9
    assert result["type"] == "heuristic"


def test_uploaded_csv_and_smi_formats():
    assert parse_uploaded_smiles("smiles,name\nCCO,ethanol\n") == ["CCO"]
    assert parse_uploaded_smiles("CCO ethanol\nCCC propane\n") == ["CCO", "CCC"]
