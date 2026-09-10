"""DrugAI Lite — computational drug-discovery screening API.

Enhanced v2 features:
- RDKit molecular descriptors + physicochemical properties
- Solubility and Tox21 ensemble predictions
- Model-disagreement uncertainty
- Lipinski + Veber drug-likeness rules
- PAINS structural-alert screening
- Morgan fingerprint nearest-neighbour analog search
- Murcko scaffold identification
- Batch screening with applicability/domain and diversity signals
- Reproducible model metadata

This is a research/portfolio prototype, not a clinical or regulatory system.
"""
import base64
import csv
import io
import json
import math
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator

BASE = Path(__file__).parent
MODELS = BASE / "models"
DATA = BASE / "data"

app = FastAPI(title="DrugAI Lite", version="2.0.0", description="Explainable computational drug-discovery screening prototype")

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

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# PAINS catalog is an RDKit structural-alert filter, not an ML prediction.
pains_params = FilterCatalogParams()
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
pains_catalog = FilterCatalog(pains_params)


def parse_mol(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


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
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "ExactMolWt": round(Descriptors.ExactMolWt(mol), 4),
        "NumRings": rdMolDescriptors.CalcNumRings(mol),
    }


def descriptor_vector(desc: dict) -> np.ndarray:
    return np.array([[desc[n] for n in DESCRIPTOR_NAMES]], dtype=float)


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
    return {"pass": len(violations) <= 1, "violations": violations, "n_violations": len(violations)}


def veber_verdict(mol) -> dict:
    rot = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    violations = []
    if rot > 10:
        violations.append("Rotatable bonds > 10")
    if tpsa > 140:
        violations.append("TPSA > 140 Å²")
    return {"pass": not violations, "violations": violations}


def structural_alerts(mol) -> dict:
    matches = pains_catalog.GetMatches(mol)
    names = []
    for match in matches:
        try:
            names.append(match.GetDescription())
        except Exception:
            names.append("PAINS alert")
    names = list(dict.fromkeys(names))
    return {"pass": not names, "count": len(names), "alerts": names[:10]}


def molecular_identity(mol) -> dict:
    formula = rdMolDescriptors.CalcMolFormula(mol)
    canonical = Chem.MolToSmiles(mol, canonical=True)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else ""
    return {
        "canonical_smiles": canonical,
        "formula": formula,
        "murcko_scaffold_smiles": scaffold_smiles,
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(mol),
    }


def fingerprint(mol):
    return morgan_generator.GetFingerprint(mol)


def predict_solubility(desc: dict) -> dict:
    X = sol_scaler.transform(descriptor_vector(desc))
    p_rf = float(sol_rf.predict(X)[0])
    p_gb = float(sol_gb.predict(X)[0])
    mean = (p_rf + p_gb) / 2
    disagreement = abs(p_rf - p_gb)
    confidence = "HIGH" if disagreement < 0.4 else ("MODERATE" if disagreement < 1.0 else "LOW")
    label = "GOOD" if mean > -2 else ("MODERATE" if mean > -4 else "POOR")
    return {
        "log_solubility_mol_per_L": round(mean, 2), "label": label,
        "model_disagreement": round(disagreement, 3), "confidence": confidence,
        "rf_prediction": round(p_rf, 2), "gb_prediction": round(p_gb, 2),
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
        "toxicity_probability": round(mean, 3), "label": label,
        "model_disagreement": round(disagreement, 3), "confidence": confidence,
        "assay": METRICS["toxicity"]["assay"] + " (androgen receptor nuclear signalling)",
        "rf_prediction": round(p_rf, 3), "gb_prediction": round(p_gb, 3),
    }


def explain(desc: dict) -> dict:
    def ranked(model):
        importances = getattr(model, "feature_importances_", np.zeros(len(DESCRIPTOR_NAMES)))
        return [{"descriptor": n, "value": desc[n], "importance": round(float(i), 4)}
                for n, i in sorted(zip(DESCRIPTOR_NAMES, importances), key=lambda x: -x[1])[:6]]
    return {"toxicity": ranked(tox_rf), "solubility": ranked(sol_rf)}


def structure_svg_b64(mol) -> str:
    from rdkit.Chem.Draw import rdMolDraw2D
    d = rdMolDraw2D.MolDraw2DSVG(360, 280)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    return base64.b64encode(d.GetDrawingText().encode()).decode()


def load_reference_library():
    refs = []
    for filename, label in [("delaney.csv", "ESOL"), ("tox21.csv", "Tox21")]:
        path = DATA / filename
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smi = (row.get("smiles") or "").strip()
                mol = parse_mol(smi)
                if mol is not None:
                    refs.append((smi, label, fingerprint(mol)))
    return refs


REFERENCE_LIBRARY = load_reference_library()


def nearest_analogs(mol, limit=5):
    fp = fingerprint(mol)
    scored = []
    for smi, source, ref_fp in REFERENCE_LIBRARY:
        score = DataStructs.TanimotoSimilarity(fp, ref_fp)
        scored.append((score, smi, source))
    scored.sort(reverse=True)
    return [{"smiles": s, "source": src, "tanimoto": round(score, 3)} for score, s, src in scored[:limit]]


def applicability_domain(mol, analogs) -> dict:
    if not analogs:
        return {"nearest_tanimoto": None, "status": "UNKNOWN", "note": "No reference molecules available."}
    sim = analogs[0]["tanimoto"]
    status = "HIGH" if sim >= 0.7 else ("MODERATE" if sim >= 0.4 else "LOW")
    return {
        "nearest_tanimoto": sim,
        "status": status,
        "note": "Higher nearest-neighbour similarity generally means the molecule is closer to the reference chemical space used by this prototype.",
    }


def composite_score(sol, tox, qed, pains_pass, lipinski_pass, veber_pass, domain_status):
    sol_component = {"GOOD": 1.0, "MODERATE": 0.6, "POOR": 0.2}[sol["label"]]
    domain_component = {"HIGH": 1.0, "MODERATE": 0.8, "LOW": 0.5, "UNKNOWN": 0.6}[domain_status]
    alert_component = 1.0 if pains_pass else 0.55
    rule_component = (1.0 if lipinski_pass else 0.7) * (1.0 if veber_pass else 0.85)
    score = (0.28 * (1 - tox["toxicity_probability"]) +
             0.25 * sol_component + 0.17 * qed +
             0.15 * domain_component + 0.10 * alert_component + 0.05 * rule_component)
    return round(float(np.clip(score, 0, 1)), 3)


def full_analysis(smiles: str) -> dict:
    mol = parse_mol(smiles)
    if mol is None:
        return {"valid": False, "smiles": smiles, "error": "Could not parse SMILES"}
    desc = compute_descriptors(mol)
    sol = predict_solubility(desc)
    tox = predict_toxicity(desc)
    lip = lipinski_verdict(desc)
    veber = veber_verdict(mol)
    alerts = structural_alerts(mol)
    identity = molecular_identity(mol)
    analogs = nearest_analogs(mol)
    domain = applicability_domain(mol, analogs)
    overall = composite_score(sol, tox, desc["QED"], alerts["pass"], lip["pass"], veber["pass"], domain["status"])
    return {
        "valid": True, "smiles": smiles, "identity": identity, "descriptors": desc,
        "solubility": sol, "toxicity": tox, "lipinski": lip, "veber": veber,
        "structural_alerts": alerts, "applicability_domain": domain,
        "nearest_analogs": analogs, "explanation": explain(desc),
        "overall_score": overall, "structure_svg_b64": structure_svg_b64(mol),
    }


class MoleculeRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=5000)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version, "reference_library_size": len(REFERENCE_LIBRARY), "metrics": METRICS}


@app.get("/api/model-cards")
def model_cards():
    return {"version": app.version, "metrics": METRICS, "features": [
        "RDKit descriptors", "solubility ensemble", "Tox21 NR-AR ensemble",
        "Lipinski + Veber filters", "PAINS alerts", "Morgan/Tanimoto analog search",
        "Murcko scaffold", "applicability-domain signal", "descriptor-importance explanation",
    ]}


@app.post("/api/analyze")
def analyze(req: MoleculeRequest):
    result = full_analysis(req.smiles.strip())
    if not result["valid"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


def parse_uploaded_smiles(content: str):
    content = content.lstrip("\ufeff")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if lines[0].lower().startswith("smiles,") or "smiles" in [x.strip().lower() for x in lines[0].split(",")]:
        reader = csv.DictReader(io.StringIO(content))
        if not reader.fieldnames or "smiles" not in [h.lower() for h in reader.fieldnames]:
            raise HTTPException(status_code=400, detail="CSV must contain a 'smiles' column.")
        smiles_key = next(h for h in reader.fieldnames if h.lower() == "smiles")
        return [(row.get(smiles_key) or "").strip() for row in reader if (row.get(smiles_key) or "").strip()]
    return [line.split()[0] for line in lines]


@app.post("/api/screen")
async def screen(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".csv", ".txt", ".smi")):
        raise HTTPException(status_code=400, detail="Upload a .csv, .txt, or .smi file.")
    content = (await file.read()).decode(errors="ignore")
    raw_smiles = parse_uploaded_smiles(content)
    if len(raw_smiles) > 5000:
        raise HTTPException(status_code=413, detail="Maximum batch size is 5,000 molecules.")

    total = len(raw_smiles)
    valid_results, invalid_count, lipinski_fail, pains_fail = [], 0, 0, 0
    seen = set()

    for smi in raw_smiles:
        mol = parse_mol(smi)
        if mol is None:
            invalid_count += 1
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in seen:
            continue
        seen.add(canonical)
        desc = compute_descriptors(mol)
        lip = lipinski_verdict(desc)
        if not lip["pass"]:
            lipinski_fail += 1
            continue
        veber = veber_verdict(mol)
        alerts = structural_alerts(mol)
        if not alerts["pass"]:
            pains_fail += 1
        sol = predict_solubility(desc)
        tox = predict_toxicity(desc)
        analogs = nearest_analogs(mol, limit=1)
        domain = applicability_domain(mol, analogs)
        overall = composite_score(sol, tox, desc["QED"], alerts["pass"], lip["pass"], veber["pass"], domain["status"])
        valid_results.append({
            "smiles": smi, "canonical_smiles": canonical, "descriptors": desc,
            "solubility": sol, "toxicity": tox, "veber": veber,
            "structural_alerts": alerts, "applicability_domain": domain,
            "overall_score": overall,
        })

    valid_results.sort(key=lambda r: (-r["overall_score"], r["toxicity"]["toxicity_probability"]))
    top = valid_results[:20]
    for r in top:
        r["structure_svg_b64"] = structure_svg_b64(parse_mol(r["smiles"]))

    funnel = {
        "submitted": total,
        "valid_structures": total - invalid_count,
        "unique_valid_structures": len(seen),
        "passed_lipinski": total - invalid_count - lipinski_fail,
        "paints_flagged": pains_fail,
        "scored": len(valid_results),
        "top_candidates": len(top),
    }
    return JSONResponse({"funnel": funnel, "candidates": top})


app.mount("/", StaticFiles(directory=str(BASE / "static"), html=True), name="static")
