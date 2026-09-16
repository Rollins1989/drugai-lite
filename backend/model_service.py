"""Model loading, prediction, metadata, and model-agreement utilities."""
import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import HTTPException

from chemistry import MORGAN, descriptor_vector, parse_mol
from config import MODELS_DIR


def _load(path):
    if not path.exists():
        raise RuntimeError(f"Required model artifact is missing: {path}")
    return joblib.load(path)


def _load_json(path, default=None):
    if not path.exists(): return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))

sol_rf = _load(MODELS_DIR / "solubility_rf.joblib")
sol_gb = _load(MODELS_DIR / "solubility_gb.joblib")
sol_scaler = _load(MODELS_DIR / "solubility_scaler.joblib")
tox_rf = _load(MODELS_DIR / "toxicity_rf.joblib")
tox_gb = _load(MODELS_DIR / "toxicity_gb.joblib")
tox_scaler = _load(MODELS_DIR / "toxicity_scaler.joblib")
METRICS = _load_json(MODELS_DIR / "metrics.json")
MODEL_VERSION = _load_json(MODELS_DIR / "model_version.json", {"version": "unknown"})

TARGET_MODELS = {}
TARGET_DIR = MODELS_DIR / "targets"
if TARGET_DIR.exists():
    for target_dir in TARGET_DIR.iterdir():
        if not target_dir.is_dir(): continue
        required = [target_dir / "rf.joblib", target_dir / "gb.joblib", target_dir / "scaler.joblib"]
        if all(p.exists() for p in required):
            TARGET_MODELS[target_dir.name] = {
                "rf": _load(required[0]), "gb": _load(required[1]), "scaler": _load(required[2]),
                "metrics": _load_json(target_dir / "metrics.json"),
            }


def agreement_signal(disagreement: float, thresholds: tuple[float, float]) -> dict:
    low, moderate = thresholds
    level = "HIGH_AGREEMENT" if disagreement < low else ("MODERATE_AGREEMENT" if disagreement < moderate else "LOW_AGREEMENT")
    return {
        "absolute_disagreement": round(float(disagreement), 4),
        "level": level,
        "method": "Absolute difference between RF and Gradient Boosting predictions.",
        "interpretation": "Lower disagreement indicates greater agreement between fitted models; it is not a calibrated probability of correctness.",
    }


def predict_solubility(desc: dict) -> dict:
    X = sol_scaler.transform(descriptor_vector(desc))
    rf, gb = float(sol_rf.predict(X)[0]), float(sol_gb.predict(X)[0])
    mean = (rf + gb) / 2
    return {
        "log_solubility_mol_per_L": round(mean, 2),
        "label": "GOOD" if mean > -2 else ("MODERATE" if mean > -4 else "POOR"),
        "model_agreement": agreement_signal(abs(rf - gb), (0.4, 1.0)),
        "rf_prediction": round(rf, 2), "gb_prediction": round(gb, 2),
        "label_definition": "GOOD > -2; MODERATE -4 to -2; POOR < -4.",
    }


def predict_toxicity(desc: dict) -> dict:
    X = tox_scaler.transform(descriptor_vector(desc))
    rf, gb = float(tox_rf.predict_proba(X)[0][1]), float(tox_gb.predict_proba(X)[0][1])
    mean = (rf + gb) / 2
    return {
        "toxicity_probability": round(mean, 3),
        "label": "LOW" if mean < 0.3 else ("MODERATE" if mean < 0.6 else "HIGH"),
        "model_agreement": agreement_signal(abs(rf - gb), (0.1, 0.25)),
        "assay": METRICS["toxicity"]["assay"] + " (androgen receptor nuclear signalling)",
        "rf_prediction": round(rf, 3), "gb_prediction": round(gb, 3),
        "label_definition": "LOW < 0.30; MODERATE 0.30–0.60; HIGH >= 0.60.",
        "note": "Probability refers to the modeled Tox21 NR-AR assay endpoint, not general human toxicity.",
    }


def explain(desc: dict) -> dict:
    def ranked(model):
        importances = getattr(model, "feature_importances_", np.zeros(12))
        return [{"descriptor": n, "value": desc[n], "importance": round(float(i), 4)} for n, i in sorted(zip(desc.keys(), importances), key=lambda x: -x[1])[:6] if n in desc]
    # Restrict explanations to the features used by the deployed models.
    names = ["MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "NumAromaticRings", "RingCount", "FractionCSP3", "NumHeteroatoms", "QED", "NumValenceElectrons"]
    def ranked_model(model):
        values = getattr(model, "feature_importances_", np.zeros(len(names)))
        return [{"descriptor": n, "value": desc[n], "importance": round(float(i), 4)} for n, i in sorted(zip(names, values), key=lambda x: -x[1])[:6]]
    return {"toxicity": ranked_model(tox_rf), "solubility": ranked_model(sol_rf), "method": "Tree-model feature_importances_; global model-level importance, not per-molecule causal attribution."}


def screening_score(sol, tox, qed, pains_pass, lipinski_pass, veber_pass, domain_status):
    sol_component = {"GOOD": 1.0, "MODERATE": 0.6, "POOR": 0.2}[sol["label"]]
    domain_component = {"HIGH": 1.0, "MODERATE": 0.8, "LOW": 0.5, "UNKNOWN": 0.6}[domain_status]
    alert_component = 1.0 if pains_pass else 0.55
    rule_component = (1.0 if lipinski_pass else 0.7) * (1.0 if veber_pass else 0.85)
    components = {
        "toxicity": 0.28 * (1 - tox["toxicity_probability"]), "solubility": 0.25 * sol_component,
        "qed": 0.17 * qed, "chemical_space_proximity": 0.15 * domain_component,
        "pains": 0.10 * alert_component, "drug_likeness_rules": 0.05 * rule_component,
    }
    return {"value": round(float(np.clip(sum(components.values()), 0, 1)), 3),
            "components": {k: round(float(v), 4) for k, v in components.items()},
            "weights": {"toxicity": 0.28, "solubility": 0.25, "qed": 0.17, "chemical_space_proximity": 0.15, "pains": 0.10, "drug_likeness_rules": 0.05},
            "type": "heuristic", "note": "Not a validated efficacy, safety, developability, or clinical score. Weights are hand-specified for portfolio screening."}


def target_activity_prediction(smiles: str, target_name: str) -> dict:
    bundle = TARGET_MODELS.get(target_name)
    if bundle is None:
        raise HTTPException(status_code=404, detail=f"No trained target model named '{target_name}'. Available: {sorted(TARGET_MODELS)}. Run train_target_activity.py first.")
    mol = parse_mol(smiles)
    if mol is None: raise HTTPException(status_code=400, detail="Could not parse SMILES.")
    vector = np.asarray(MORGAN.GetFingerprintAsNumPy(mol), dtype=np.uint8).reshape(1, -1)
    X = bundle["scaler"].transform(vector)
    rf, gb = float(bundle["rf"].predict(X)[0]), float(bundle["gb"].predict(X)[0])
    pred = (rf + gb) / 2
    return {"target": target_name, "predicted_pIC50": round(pred, 3), "predicted_IC50_nM": round(float(10 ** (9 - pred)), 2),
            "rf_pIC50": round(rf, 3), "gb_pIC50": round(gb, 3), "model_agreement": agreement_signal(abs(rf-gb), (0.35, 0.75)),
            "pIC50_threshold_6_exceeded": bool(pred >= 6.0), "training_evaluation": bundle["metrics"],
            "note": "pIC50 is a model prediction, not experimental activity. The pIC50=6 flag is a screening threshold, not an efficacy claim."}
