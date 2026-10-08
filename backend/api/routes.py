"""Versioned FastAPI API for DrugAI Lite."""
from __future__ import annotations
import csv, io, uuid
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from chemistry import REFERENCE_LIBRARY, set_custom_reference_library
from config import APP_VERSION, MAX_UPLOAD_BYTES
from model_service import METRICS, MODEL_VERSION, TARGET_MODELS
from schemas import ActivityRequest, ChemicalSpaceRequest, CompareRequest, MoleculeRequest, ParetoRequest
from services import full_analysis, parse_uploaded_records, screen_records, start_screen_job, get_screen_job
from analytics import chemical_space, dataset_quality, pairwise_similarity, pareto_rank
from provenance import runtime_provenance
from reports import molecule_pdf

router=APIRouter(prefix="/api")

@router.get("/health")
def health():return {"status":"ok","version":APP_VERSION,"model_version":MODEL_VERSION,"reference_library_size":len(REFERENCE_LIBRARY),"targets":sorted(TARGET_MODELS)}
@router.get("/version")
def version():return runtime_provenance(APP_VERSION,MODEL_VERSION)
@router.get("/model-cards")
def model_cards():
    return {"version":APP_VERSION,"model_version":MODEL_VERSION,"metrics":METRICS,
            "features":["molecular profiling","ESOL solubility ensemble","Tox21 NR-AR ensemble","target-specific activity","applicability-domain signal","empirical ensemble spread","local sensitivity diagnostics","Morgan/Tanimoto analog search","scaffold analysis","Pareto multi-objective ranking","chemical-space PCA/clustering","dataset quality reports","CSV/PDF exports","Docker/CI"],
            "interpretation":{"model_agreement":"RF/Gradient Boosting disagreement; not calibrated confidence.","uncertainty":"Empirical RF-tree 10th–90th percentile spread; not a calibrated prediction interval.","screening_score":"Hand-weighted heuristic candidate-prioritization score.","applicability_domain":"Nearest-neighbour chemical-space proximity signal."},
            "limitations":["No external or prospective experimental validation is claimed.","Tox21 NR-AR is one assay endpoint, not overall human toxicity.","Tree importance/local sensitivity are diagnostics, not causal attribution.","ChEMBL activity data contain assay and measurement heterogeneity.","All predictions are hypothesis-generation outputs requiring experimental validation."]}

@router.post("/analyze")
def analyze(req:MoleculeRequest,target:str|None=None):
    result=full_analysis(req.smiles.strip(),target.strip().lower() if target else None)
    if not result["valid"]:raise HTTPException(status_code=400,detail=result["error"])
    return result

@router.post("/compare")
def compare(req:CompareRequest):
    a=full_analysis(req.smiles_a.strip()); b=full_analysis(req.smiles_b.strip())
    if not a["valid"] or not b["valid"]:raise HTTPException(status_code=400,detail="Both SMILES must be valid.")
    return {"molecule_a":a,"molecule_b":b,"structural_similarity":pairwise_similarity(req.smiles_a,req.smiles_b)}

@router.post("/chemical-space")
def chemical_space_route(req:ChemicalSpaceRequest):
    if req.labels and len(req.labels)!=len(req.smiles):raise HTTPException(status_code=422,detail="labels must match smiles length.")
    return chemical_space(req.smiles,req.labels,req.n_clusters)

@router.post("/pareto")
def pareto(req:ParetoRequest):
    allowed={"activity","toxicity","solubility","qed","mw","logp"}
    if not set(req.objectives)<=allowed:raise HTTPException(status_code=422,detail=f"Unknown objectives. Use: {sorted(allowed)}")
    return {"objectives":req.objectives,"rows":pareto_rank(req.rows,req.objectives)}

@router.post("/dataset/quality")
async def dataset_quality_route(file:UploadFile=File(...)):
    content=await file.read()
    if len(content)>MAX_UPLOAD_BYTES:raise HTTPException(status_code=413,detail="Uploaded file is too large.")
    return dataset_quality(parse_uploaded_records(content.decode(errors="replace")))

@router.post("/references/upload")
async def upload_reference(file:UploadFile=File(...)):
    content=await file.read()
    if len(content)>MAX_UPLOAD_BYTES: raise HTTPException(status_code=413,detail="Uploaded reference library is too large.")
    records=parse_uploaded_records(content.decode(errors="replace"))
    count=set_custom_reference_library(records)
    return {"loaded_molecules":count,"reference_type":"custom","note":"Custom reference libraries are process-local and reset when the server restarts."}

@router.get("/activity-targets")
def activity_targets():
    return {"targets":[{"name":name,"target_chembl_id":bundle["metrics"].get("target_chembl_id"),"evaluation":bundle["metrics"].get("model",{})} for name,bundle in sorted(TARGET_MODELS.items())]}

@router.post("/predict-activity")
def predict_activity(req:ActivityRequest):return __import__("model_service").target_activity_prediction(req.smiles.strip(),req.target.strip().lower())

@router.post("/screen")
async def screen(file:UploadFile=File(...),target:str|None=Form(None)):
    if not file.filename or not file.filename.lower().endswith((".csv",".txt",".smi")):raise HTTPException(status_code=400,detail="Upload a .csv, .txt, or .smi file.")
    content=await file.read()
    if len(content)>MAX_UPLOAD_BYTES:raise HTTPException(status_code=413,detail="Uploaded file is too large (10 MB maximum).")
    return JSONResponse(screen_records(parse_uploaded_records(content.decode(errors="replace")),target.strip().lower() if target else None))

@router.post("/screen/jobs")
async def screen_job(file:UploadFile=File(...),target:str|None=Form(None)):
    content=await file.read()
    if len(content)>MAX_UPLOAD_BYTES:raise HTTPException(status_code=413,detail="Uploaded file is too large.")
    return start_screen_job(parse_uploaded_records(content.decode(errors="replace")),target.strip().lower() if target else None)

@router.get("/screen/jobs/{job_id}")
def screen_job_status(job_id:str):return get_screen_job(job_id)

@router.post("/screen/export.csv")
def export_screen_csv(payload:dict):
    rows=payload.get("candidates",[]); out=io.StringIO(); fields=["rank","smiles","canonical_smiles","predicted_pIC50","toxicity_probability","solubility","qed","mw","logp","nearest_tanimoto","domain_status","screening_score","pareto_optimal"]
    writer=csv.DictWriter(out,fieldnames=fields); writer.writeheader()
    for i,r in enumerate(rows,1):
        d=r.get("descriptors",{}); writer.writerow({"rank":i,"smiles":r.get("smiles"),"canonical_smiles":r.get("canonical_smiles"),"predicted_pIC50":r.get("predicted_pIC50"),"toxicity_probability":r.get("toxicity",{}).get("toxicity_probability"),"solubility":r.get("solubility",{}).get("log_solubility_mol_per_L"),"qed":d.get("QED"),"mw":d.get("MolWt"),"logp":d.get("LogP"),"nearest_tanimoto":r.get("applicability_domain",{}).get("nearest_tanimoto"),"domain_status":r.get("applicability_domain",{}).get("status"),"screening_score":r.get("screening_score",{}).get("value"),"pareto_optimal":r.get("pareto_optimal")})
    return Response(content=out.getvalue(),media_type="text/csv",headers={"Content-Disposition":"attachment; filename=drugai_screening_results.csv"})

@router.post("/analyze/report.pdf")
def report_pdf(req:MoleculeRequest,target:str|None=None):
    result=full_analysis(req.smiles.strip(),target.strip().lower() if target else None)
    if not result["valid"]:raise HTTPException(status_code=400,detail=result["error"])
    pdf=molecule_pdf(result,runtime_provenance(APP_VERSION,MODEL_VERSION))
    return Response(content=pdf,media_type="application/pdf",headers={"Content-Disposition":"attachment; filename=drugai_molecular_report.pdf"})
