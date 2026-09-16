"""FastAPI route handlers. Business logic lives in services/model_service."""
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from chemistry import REFERENCE_LIBRARY
from config import APP_VERSION, MAX_BATCH_MOLECULES
from model_service import METRICS, MODEL_VERSION, TARGET_MODELS, target_activity_prediction
from schemas import ActivityRequest, MoleculeRequest
from services import full_analysis, parse_uploaded_smiles, screen_library

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION, "model_version": MODEL_VERSION,
            "reference_library_size": len(REFERENCE_LIBRARY), "metrics": METRICS, "targets": sorted(TARGET_MODELS)}


@router.get("/version")
def version():
    return {"application_version": APP_VERSION, "model_version": MODEL_VERSION}


@router.get("/model-cards")
def model_cards():
    return {"version": APP_VERSION, "model_version": MODEL_VERSION, "metrics": METRICS,
            "features": ["RDKit descriptors", "solubility ensemble", "Tox21 NR-AR ensemble", "Lipinski + Veber filters", "PAINS alerts", "Morgan/Tanimoto analog search", "Murcko scaffold", "chemical-space proximity signal", "global descriptor-importance explanation", "random + scaffold evaluation", "target-specific pIC50 prediction"],
            "interpretation": {"model_agreement": "RF/GB prediction disagreement; not calibrated confidence.", "screening_score": "Hand-weighted heuristic prioritization score; not a validated scientific endpoint.", "applicability_domain": "Nearest-neighbour chemical-space proximity signal."},
            "limitations": ["No external validation or prospective experimental validation is claimed.", "Tree-model feature importance is global, not a causal per-molecule explanation.", "Tox21 NR-AR is one assay endpoint and should not be generalized to overall human toxicity.", "ChEMBL target activity data are observational experimental measurements with assay heterogeneity.", "Predictions should be treated as hypothesis-generation outputs only."]}


@router.post("/analyze")
def analyze(req: MoleculeRequest):
    result = full_analysis(req.smiles.strip())
    if not result["valid"]: raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/activity-targets")
def activity_targets():
    return {"targets": [{"name": name, "target_chembl_id": bundle["metrics"].get("target_chembl_id"), "evaluation": bundle["metrics"].get("model", {})} for name, bundle in sorted(TARGET_MODELS.items())]}


@router.post("/predict-activity")
def predict_activity(req: ActivityRequest):
    return target_activity_prediction(req.smiles.strip(), req.target.strip().lower())


@router.post("/screen")
async def screen(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".csv", ".txt", ".smi")):
        raise HTTPException(status_code=400, detail="Upload a .csv, .txt, or .smi file.")
    content = (await file.read()).decode(errors="ignore")
    if len(content.encode()) > 10_000_000:
        raise HTTPException(status_code=413, detail="Uploaded file is too large (10 MB maximum).")
    return JSONResponse(screen_library(parse_uploaded_smiles(content)))
