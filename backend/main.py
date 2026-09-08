"""
DrugAI Lite — backend API.

Real, working slice of the larger DrugAI vision:
  - RDKit molecular descriptor computation
  - Trained ensemble models (solubility regression, toxicity classification)
  - Model-disagreement uncertainty (not simulated)
  - Lipinski / QED drug-likeness scoring
  - 2D structure rendering
  - Batch virtual screening funnel (validity -> Lipinski -> ML scoring -> ranking)
  - Descriptor-importance explainability

No docking, no GNN, no knowledge graph, no auth/billing in this build —
see README "Roadmap" for the honest boundary of what's real here.
"""
import io
import json
import base64
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, Lipinski, Draw

BASE = Path(__file__).parent
MODELS = BASE / "models"

app = FastAPI(title="DrugAI Lite")

# ---------------------------------------------------------------- models --
sol_rf = joblib.load(MODELS / "solubility_rf.joblib")
sol_gb = joblib.load(MODELS / "solubility_gb.joblib")
sol_scaler = joblib.load(MODELS / "solubility_scaler.joblib")

tox_rf = joblib.load(MODELS / "toxicity_rf.joblib")
tox_gb = joblib.load(MODELS / "toxicity_gb.joblib")
tox_scaler = joblib.load(MODELS / "toxicity_scaler.joblib")

METRICS = json.loads((MODELS / "metrics.json").read_text())

DESCRIPTOR_NAMES = [
    "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "RingCount",
    "FractionCSP3", "NumHeteroatoms", "QED", "NumValenceElectrons",
]


# ------------------------------------------------------------- chemistry --
def parse_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return mol


def compute_descriptors(mol) -> dict:
    return {
        "MolWt": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Crippen.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": Lipinski.NumAromaticRings(mol),
        "RingCount": Descriptors.RingCount(mol),
        "FractionCSP3": round(Descriptors.FractionCSP3(mol), 3),
        "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),
        "QED": round(QED.qed(mol), 3),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
    }


def descriptor_vector(desc: dict) -> np.ndarray:
    return np.array([[desc[n] for n in DESCRIPTOR_NAMES]])


def lipinski_verdict(desc: dict) -> dict:
    violations = []
    if desc["MolWt"] > 500:
        violations.append("Molecular weight > 500")
    if desc["LogP"] > 5:
        violations.append("LogP > 5")
    if desc["NumHDonors"] > 5:
        violations.append("H-bond donors > 5")
    if desc["NumHAcceptors"] > 10:
        violations.append("H-bond acceptors > 10")
    return {
        "pass": len(violations) <= 1,  # Lipinski allows 1 violation
        "violations": violations,
        "n_violations": len(violations),
    }


def predict_solubility(desc: dict) -> dict:
    X = sol_scaler.transform(descriptor_vector(desc))
    p_rf = float(sol_rf.predict(X)[0])
    p_gb = float(sol_gb.predict(X)[0])
    mean = (p_rf + p_gb) / 2
    disagreement = abs(p_rf - p_gb)
    confidence = "HIGH" if disagreement < 0.4 else ("MODERATE" if disagreement < 1.0 else "LOW")
    label = "GOOD" if mean > -2 else ("MODERATE" if mean > -4 else "POOR")
    return {
        "log_solubility_mol_per_L": round(mean, 2),
        "label": label,
        "model_disagreement": round(disagreement, 3),
        "confidence": confidence,
        "rf_prediction": round(p_rf, 2),
        "gb_prediction": round(p_gb, 2),
    }


def predict_toxicity(desc: dict) -> dict:
    X = tox_scaler.transform(descriptor_vector(desc))
    p_rf = float(tox_rf.predict_proba(X)[0][1])
    p_gb = float(tox_gb.predict_proba(X)[0][1])
    mean = (p_rf + p_gb) / 2
    disagreement = abs(p_rf - p_gb)
    confidence = "HIGH" if disagreement < 0.1 else ("MODERATE" if disagreement < 0.25 else "LOW")
    label = "LOW" if mean < 0.3 else ("MODERATE" if mean < 0.6 else "HIGH")
    return {
        "toxicity_probability": round(mean, 3),
        "label": label,
        "model_disagreement": round(disagreement, 3),
        "confidence": confidence,
        "assay": METRICS["toxicity"]["assay"] + " (androgen receptor nuclear signalling)",
        "rf_prediction": round(p_rf, 3),
        "gb_prediction": round(p_gb, 3),
    }


def explain(desc: dict) -> list:
    """Descriptor-importance explanation using the trained RF's feature_importances_."""
    importances = tox_rf.feature_importances_
    ranked = sorted(zip(DESCRIPTOR_NAMES, importances), key=lambda x: -x[1])[:4]
    notes = []
    for name, imp in ranked:
        notes.append(f"{name} = {desc[name]} (model weight {imp:.2f})")
    return notes


def structure_svg_b64(mol) -> str:
    from rdkit.Chem.Draw import rdMolDraw2D
    d = rdMolDraw2D.MolDraw2DSVG(320, 260)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    return base64.b64encode(svg.encode()).decode()


def full_analysis(smiles: str) -> dict:
    mol = parse_mol(smiles)
    if mol is None:
        return {"valid": False, "smiles": smiles, "error": "Could not parse SMILES"}
    desc = compute_descriptors(mol)
    sol = predict_solubility(desc)
    tox = predict_toxicity(desc)
    lip = lipinski_verdict(desc)
    overall = round((0.35 * (1 - tox["toxicity_probability"]) +
                      0.35 * (1 if sol["label"] == "GOOD" else 0.6 if sol["label"] == "MODERATE" else 0.2) +
                      0.30 * desc["QED"]), 3)
    return {
        "valid": True,
        "smiles": smiles,
        "descriptors": desc,
        "solubility": sol,
        "toxicity": tox,
        "lipinski": lip,
        "explanation": explain(desc),
        "overall_score": overall,
        "structure_svg_b64": structure_svg_b64(mol),
    }


# ------------------------------------------------------------------ API --
class MoleculeRequest(BaseModel):
    smiles: str


@app.get("/api/health")
def health():
    return {"status": "ok", "metrics": METRICS}


@app.post("/api/analyze")
def analyze(req: MoleculeRequest):
    result = full_analysis(req.smiles.strip())
    if not result["valid"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/screen")
async def screen(file: UploadFile = File(...)):
    content = (await file.read()).decode(errors="ignore")
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    # accept plain SMILES-per-line, or CSV with a 'smiles' column
    if "," in lines[0].lower() and "smiles" in lines[0].lower():
        header = [h.strip().lower() for h in lines[0].split(",")]
        idx = header.index("smiles")
        raw_smiles = [l.split(",")[idx].strip() for l in lines[1:]]
    else:
        raw_smiles = lines

    total = len(raw_smiles)
    valid_results = []
    invalid_count = 0
    lipinski_fail = 0

    for smi in raw_smiles:
        mol = parse_mol(smi)
        if mol is None:
            invalid_count += 1
            continue
        desc = compute_descriptors(mol)
        lip = lipinski_verdict(desc)
        if not lip["pass"]:
            lipinski_fail += 1
            continue
        sol = predict_solubility(desc)
        tox = predict_toxicity(desc)
        overall = round((0.35 * (1 - tox["toxicity_probability"]) +
                          0.35 * (1 if sol["label"] == "GOOD" else 0.6 if sol["label"] == "MODERATE" else 0.2) +
                          0.30 * desc["QED"]), 3)
        valid_results.append({
            "smiles": smi,
            "descriptors": desc,
            "solubility": sol,
            "toxicity": tox,
            "overall_score": overall,
        })

    valid_results.sort(key=lambda r: -r["overall_score"])
    top = valid_results[:20]
    for r in top:
        mol = parse_mol(r["smiles"])
        r["structure_svg_b64"] = structure_svg_b64(mol)

    funnel = {
        "submitted": total,
        "valid_structures": total - invalid_count,
        "passed_lipinski": total - invalid_count - lipinski_fail,
        "scored": len(valid_results),
        "top_candidates": len(top),
    }

    return JSONResponse({"funnel": funnel, "candidates": top})


app.mount("/", StaticFiles(directory=str(BASE / "static"), html=True), name="static")
