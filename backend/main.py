"""DrugAI Lite — computational drug-discovery screening API.

Research/portfolio prototype for molecular property prediction and virtual screening.

Important interpretation notes:
- Ensemble disagreement is reported as a model-agreement signal, not calibrated confidence.
- The screening score is a heuristic prioritization score, not a validated efficacy/safety score.
- Applicability-domain status is a nearest-neighbour chemical-space proximity signal.
"""
import base64
import csv
import io
import json
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator

BASE = Path(__file__).parent
MODELS = BASE / "models"
DATA = BASE / "data"
APP_VERSION = "2.1.0"

app = FastAPI(
    title="DrugAI Lite",
    version=APP_VERSION,
    description="Explainable computational drug-discovery screening prototype",
)

sol_rf = joblib.load(MODELS / "solubility_rf.joblib")
sol_gb = joblib.load(MODELS / "solubility_gb.joblib")
sol_scaler = joblib.load(MODELS / "solubility_scaler.joblib")
tox_rf = joblib.load(MODELS / "toxicity_rf.joblib")
tox_gb = joblib.load(MODELS / "toxicity_gb.joblib")
tox_scaler = joblib.load(MODELS / "toxicity_scaler.joblib")
METRICS = json.loads((MODELS / "metrics.json").read_text())

TARGET_MODELS = {}
TARGET_MODEL_DIR = MODELS / "targets"
if TARGET_MODEL_DIR.exists():
    for target_dir in TARGET_MODEL_DIR.iterdir():
        if not target_dir.is_dir():
            continue
        required = [target_dir / "rf.joblib", target_dir / "gb.joblib", target_dir / "scaler.joblib"]
        if all(p.exists() for p in required):
            TARGET_MODELS[target_dir.name] = {
                "rf": joblib.load(required[0]),
                "gb": joblib.load(required[1]),
                "scaler": joblib.load(required[2]),
                "metrics": json.loads((target_dir / "metrics.json").read_text())
                if (target_dir / "metrics.json").exists() else {},
            }

DESCRIPTOR_NAMES = [
    "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "RingCount",
    "FractionCSP3", "NumHeteroatoms", "QED", "NumValenceElectrons",
]
morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

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
    violations = []
    if Descriptors.NumRotatableBonds(mol) > 10:
        violations.append("Rotatable bonds > 10")
    if Descriptors.TPSA(mol) > 140:
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


def agreement_signal(disagreement: float, thresholds: tuple[float, float]) -> dict:
    low, moderate = thresholds
    level = "HIGH_AGREEMENT" if disagreement < low else (
        "MODERATE_AGREEMENT" if disagreement < moderate else "LOW_AGREEMENT"
    )
    return {
        "absolute_disagreement": round(float(disagreement), 4),
        "level": level,
        "method": "Absolute difference between RF and Gradient Boosting predictions.",
        "interpretation": "Lower disagreement indicates greater agreement between fitted models; it is not a calibrated probability of correctness.",
    }


def target_activity_prediction(smiles: str, target_name: str) -> dict:
    model_bundle = TARGET_MODELS.get(target_name)
    if model_bundle is None:
        available = sorted(TARGET_MODELS)
        raise HTTPException(status_code=404, detail=f"No trained target model named '{target_name}'. Available: {available}. Run train_target_activity.py first.")
    mol = parse_mol(smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Could not parse SMILES.")
    vector = np.asarray(morgan_generator.GetFingerprintAsNumPy(mol), dtype=np.uint8).reshape(1, -1)
    X = model_bundle["scaler"].transform(vector)
    p_rf = float(model_bundle["rf"].predict(X)[0])
    p_gb = float(model_bundle["gb"].predict(X)[0])
    pred = (p_rf + p_gb) / 2
    disagreement = abs(p_rf - p_gb)
    return {
        "target": target_name,
        "predicted_pIC50": round(pred, 3),
        "predicted_IC50_nM": round(float(10 ** (9 - pred)), 2),
        "rf_pIC50": round(p_rf, 3),
        "gb_pIC50": round(p_gb, 3),
        "model_agreement": agreement_signal(disagreement, (0.35, 0.75)),
        "pIC50_threshold_6_exceeded": bool(pred >= 6.0),
        "training_evaluation": model_bundle["metrics"],
        "note": "pIC50 is a model prediction, not experimental activity. The pIC50=6 flag is a screening threshold, not an efficacy claim.",
    }


def predict_solubility(desc: dict) -> dict:
    X = sol_scaler.transform(descriptor_vector(desc))
    p_rf = float(sol_rf.predict(X)[0])
    p_gb = float(sol_gb.predict(X)[0])
    mean = (p_rf + p_gb) / 2
    disagreement = abs(p_rf - p_gb)
    label = "GOOD" if mean > -2 else ("MODERATE" if mean > -4 else "POOR")
    return {
        "log_solubility_mol_per_L": round(mean, 2),
        "label": label,
        "model_agreement": agreement_signal(disagreement, (0.4, 1.0)),
        "rf_prediction": round(p_rf, 2),
        "gb_prediction": round(p_gb, 2),
        "label_definition": "GOOD > -2; MODERATE -4 to -2; POOR < -4.",
    }


def predict_toxicity(desc: dict) -> dict:
    X = tox_scaler.transform(descriptor_vector(desc))
    p_rf = float(tox_rf.predict_proba(X)[0][1])
    p_gb = float(tox_gb.predict_proba(X)[0][1])
    mean = (p_rf + p_gb) / 2
    disagreement = abs(p_rf - p_gb)
    label = "LOW" if mean < 0.3 else ("MODERATE" if mean < 0.6 else "HIGH")
    return {
        "toxicity_probability": round(mean, 3),
        "label": label,
        "model_agreement": agreement_signal(disagreement, (0.1, 0.25)),
        "assay": METRICS["toxicity"]["assay"] + " (androgen receptor nuclear signalling)",
        "rf_prediction": round(p_rf, 3),
        "gb_prediction": round(p_gb, 3),
        "label_definition": "LOW < 0.30; MODERATE 0.30–0.60; HIGH >= 0.60.",
        "note": "Probability refers to the modeled Tox21 NR-AR assay endpoint, not general human toxicity.",
    }


def explain(desc: dict) -> dict:
    def ranked(model):
        importances = getattr(model, "feature_importances_", np.zeros(len(DESCRIPTOR_NAMES)))
        return [{"descriptor": n, "value": desc[n], "importance": round(float(i), 4)} for n, i in sorted(zip(DESCRIPTOR_NAMES, importances), key=lambda x: -x[1])[:6]]
    return {
        "toxicity": ranked(tox_rf),
        "solubility": ranked(sol_rf),
        "method": "Tree-model feature_importances_; global model-level importance, not per-molecule causal attribution.",
    }


def structure_svg_b64(mol) -> str:
    from rdkit.Chem.Draw import rdMolDraw2D
    drawer = rdMolDraw2D.MolDraw2DSVG(360, 280)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return base64.b64encode(drawer.GetDrawingText().encode()).decode()


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
    scored = [(DataStructs.TanimotoSimilarity(fp, ref_fp), smi, source) for smi, source, ref_fp in REFERENCE_LIBRARY]
    scored.sort(reverse=True)
    return [{"smiles": smi, "source": source, "tanimoto": round(score, 3)} for score, smi, source in scored[:limit]]


def applicability_domain(mol, analogs) -> dict:
    if not analogs:
        return {"nearest_tanimoto": None, "status": "UNKNOWN", "note": "No reference molecules available."}
    sim = analogs[0]["tanimoto"]
    status = "HIGH" if sim >= 0.7 else ("MODERATE" if sim >= 0.4 else "LOW")
    return {
        "nearest_tanimoto": sim,
        "status": status,
        "method": "Maximum Tanimoto similarity to the bundled ESOL/Tox21 reference library.",
        "note": "This is a chemical-space proximity signal, not a formal statistical applicability-domain model or guarantee of prediction reliability.",
    }


def screening_score(sol, tox, qed, pains_pass, lipinski_pass, veber_pass, domain_status):
    """Return the explicitly heuristic candidate-prioritization score."""
    sol_component = {"GOOD": 1.0, "MODERATE": 0.6, "POOR": 0.2}[sol["label"]]
    domain_component = {"HIGH": 1.0, "MODERATE": 0.8, "LOW": 0.5, "UNKNOWN": 0.6}[domain_status]
    alert_component = 1.0 if pains_pass else 0.55
    rule_component = (1.0 if lipinski_pass else 0.7) * (1.0 if veber_pass else 0.85)
    components = {
        "toxicity": 0.28 * (1 - tox["toxicity_probability"]),
        "solubility": 0.25 * sol_component,
        "qed": 0.17 * qed,
        "chemical_space_proximity": 0.15 * domain_component,
        "pains": 0.10 * alert_component,
        "drug_likeness_rules": 0.05 * rule_component,
    }
    return {
        "value": round(float(np.clip(sum(components.values()), 0, 1)), 3),
        "components": {k: round(float(v), 4) for k, v in components.items()},
        "weights": {"toxicity": 0.28, "solubility": 0.25, "qed": 0.17, "chemical_space_proximity": 0.15, "pains": 0.10, "drug_likeness_rules": 0.05},
        "type": "heuristic",
        "note": "Not a validated efficacy, safety, developability, or clinical score. Weights are hand-specified for portfolio screening and are not empirically optimized decision thresholds.",
    }


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
    score = screening_score(sol, tox, desc["QED"], alerts["pass"], lip["pass"], veber["pass"], domain["status"])
    return {
        "valid": True,
        "smiles": smiles,
        "identity": identity,
        "descriptors": desc,
        "solubility": sol,
        "toxicity": tox,
        "lipinski": lip,
        "veber": veber,
        "structural_alerts": alerts,
        "applicability_domain": domain,
        "nearest_analogs": analogs,
        "explanation": explain(desc),
        "screening_score": score,
        "structure_svg_b64": structure_svg_b64(mol),
    }


class MoleculeRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=5000)


class ActivityRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=5000)
    target: str = Field(default="egfr", min_length=1, max_length=100)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version, "reference_library_size": len(REFERENCE_LIBRARY), "metrics": METRICS, "targets": sorted(TARGET_MODELS)}


@app.get("/api/model-cards")
def model_cards():
    return {
        "version": app.version,
        "metrics": METRICS,
        "features": ["RDKit descriptors", "solubility ensemble", "Tox21 NR-AR ensemble", "Lipinski + Veber filters", "PAINS alerts", "Morgan/Tanimoto analog search", "Murcko scaffold", "chemical-space proximity signal", "global descriptor-importance explanation", "random + scaffold evaluation", "target-specific pIC50 prediction"],
        "interpretation": {
            "model_agreement": "RF/GB prediction disagreement; not calibrated confidence.",
            "screening_score": "Hand-weighted heuristic prioritization score; not a validated scientific endpoint.",
            "applicability_domain": "Nearest-neighbour chemical-space proximity signal.",
        },
        "limitations": [
            "No external validation or prospective experimental validation is claimed.",
            "Tree-model feature importance is global, not a causal per-molecule explanation.",
            "Tox21 NR-AR is one assay endpoint and should not be generalized to overall human toxicity.",
            "ChEMBL target activity data are observational experimental measurements with assay heterogeneity.",
            "Predictions should be treated as hypothesis-generation outputs only.",
        ],
    }


@app.post("/api/analyze")
def analyze(req: MoleculeRequest):
    result = full_analysis(req.smiles.strip())
    if not result["valid"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/activity-targets")
def activity_targets():
    return {"targets": [{"name": name, "target_chembl_id": bundle["metrics"].get("target_chembl_id"), "evaluation": bundle["metrics"].get("model", {})} for name, bundle in sorted(TARGET_MODELS.items())]}


@app.post("/api/predict-activity")
def predict_activity(req: ActivityRequest):
    return target_activity_prediction(req.smiles.strip(), req.target.strip().lower())


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
    valid_results = []
    invalid_count = 0
    lipinski_fail = 0
    pains_fail = 0
    duplicate_count = 0
    seen = set()

    for smi in raw_smiles:
        mol = parse_mol(smi)
        if mol is None:
            invalid_count += 1
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in seen:
            duplicate_count += 1
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
        score = screening_score(sol, tox, desc["QED"], alerts["pass"], lip["pass"], veber["pass"], domain["status"])
        valid_results.append({"smiles": smi, "canonical_smiles": canonical, "descriptors": desc, "solubility": sol, "toxicity": tox, "veber": veber, "structural_alerts": alerts, "applicability_domain": domain, "screening_score": score})

    valid_results.sort(key=lambda r: (-r["screening_score"]["value"], r["toxicity"]["toxicity_probability"]))
    top = valid_results[:20]
    for result in top:
        result["structure_svg_b64"] = structure_svg_b64(parse_mol(result["smiles"]))

    funnel = {
        "submitted": total,
        "invalid_structures": invalid_count,
        "unique_valid_structures": len(seen),
        "duplicates_removed": duplicate_count,
        "passed_lipinski": total - invalid_count - duplicate_count - lipinski_fail,
        "paints_flagged": pains_fail,
        "scored": len(valid_results),
        "top_candidates": len(top),
    }
    return JSONResponse({"funnel": funnel, "candidates": top})


app.mount("/", StaticFiles(directory=str(BASE / "static"), html=True), name="static")
