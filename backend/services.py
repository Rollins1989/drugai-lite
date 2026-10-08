"""Domain services for molecule analysis, library ingestion, screening and jobs."""
from __future__ import annotations
import csv
import io, threading, uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException
from rdkit import Chem
from chemistry import compute_descriptors, lipinski_verdict, molecular_identity, nearest_analogs, parse_mol, structural_alerts, structure_svg_b64, veber_verdict
from config import MAX_BATCH_MOLECULES
from model_service import explain, local_sensitivity, predict_solubility, predict_toxicity, screening_score, target_activity_prediction
from analytics import dataset_quality, pareto_rank

_EXECUTOR=ThreadPoolExecutor(max_workers=2)
_JOBS={}
_LOCK=threading.Lock()

def domain_signal(analogs):
    if not analogs:return {"nearest_tanimoto":None,"status":"UNKNOWN","method":"Nearest-neighbour chemical-space signal","note":"No reference molecules available."}
    sim=analogs[0]["tanimoto"]
    return {"nearest_tanimoto":sim,"top5_mean_tanimoto":round(sum(x["tanimoto"] for x in analogs[:5])/min(5,len(analogs)),3),"status":"HIGH" if sim>=.7 else ("MODERATE" if sim>=.4 else "LOW"),"method":"Maximum Morgan/Tanimoto similarity to the selected reference library.","note":"Chemical-space proximity signal, not a formal statistical applicability-domain guarantee."}

def full_analysis(smiles:str, target:str|None=None)->dict:
    mol=parse_mol(smiles)
    if mol is None:return {"valid":False,"smiles":smiles,"error":"Could not parse SMILES"}
    desc=compute_descriptors(mol); sol=predict_solubility(desc); tox=predict_toxicity(desc)
    lip=lipinski_verdict(desc); veber=veber_verdict(mol); alerts=structural_alerts(mol)
    identity=molecular_identity(mol); analogs=nearest_analogs(mol)
    domain=domain_signal(analogs); score=screening_score(sol,tox,desc["QED"],alerts["pass"],lip["pass"],veber["pass"],domain["status"])
    result={"valid":True,"smiles":smiles,"identity":identity,"descriptors":desc,"solubility":sol,"toxicity":tox,"lipinski":lip,"veber":veber,"structural_alerts":alerts,"applicability_domain":domain,"nearest_analogs":analogs,"explanation":explain(desc),"local_sensitivity":local_sensitivity(desc),"screening_score":score,"structure_svg_b64":structure_svg_b64(mol)}
    if target:
        try: result["target_activity"]=target_activity_prediction(smiles,target)
        except HTTPException: result["target_activity"]={"error":f"Target '{target}' is not currently available."}
    return result

def parse_uploaded_records(content:str)->list[dict]:
    content=content.lstrip("\ufeff"); lines=[x.strip() for x in content.splitlines() if x.strip()]
    if not lines:raise HTTPException(status_code=400,detail="Uploaded file is empty.")
    if "," in lines[0] and "smiles" in [x.strip().lower() for x in lines[0].split(",")]:
        reader=csv.DictReader(io.StringIO(content))
        fields=reader.fieldnames or []; key=next((h for h in fields if h.lower()=="smiles"),None)
        if not key:raise HTTPException(status_code=400,detail="CSV must contain a 'smiles' column.")
        return [dict(row) for row in reader if (row.get(key) or "").strip()]
    return [{"smiles":line.split()[0]} for line in lines]

def parse_uploaded_smiles(content:str):return [r["smiles"].strip() for r in parse_uploaded_records(content)]

def _screen_records(records,target=None):
    pairs=[(str(r.get("smiles","")).strip(),r) for r in records if str(r.get("smiles","")).strip()]
    raw=[p[0] for p in pairs]
    if len(raw)>MAX_BATCH_MOLECULES:raise HTTPException(status_code=413,detail=f"Maximum batch size is {MAX_BATCH_MOLECULES:,} molecules.")
    valid=[]; invalid=duplicates=lipinski_fail=pains_flagged=0; seen=set()
    for idx,(smi,meta) in enumerate(pairs):
        mol=parse_mol(smi)
        if mol is None:invalid+=1;continue
        canonical=Chem.MolToSmiles(mol,canonical=True)
        if canonical in seen:duplicates+=1;continue
        seen.add(canonical); desc=compute_descriptors(mol); lip=lipinski_verdict(desc)
        if not lip["pass"]:lipinski_fail+=1;continue
        veber=veber_verdict(mol); alerts=structural_alerts(mol); pains_flagged+=int(not alerts["pass"])
        sol=predict_solubility(desc); tox=predict_toxicity(desc); analogs=nearest_analogs(mol,1); domain=domain_signal(analogs)
        score=screening_score(sol,tox,desc["QED"],alerts["pass"],lip["pass"],veber["pass"],domain["status"])
        row={"input":meta,"smiles":smi,"canonical_smiles":canonical,"descriptors":desc,"solubility":sol,"toxicity":tox,"veber":veber,"structural_alerts":alerts,"applicability_domain":domain,"screening_score":score}
        if target:
            try:
                row["target_activity"]=target_activity_prediction(smi,target)
                row["predicted_pIC50"]=row["target_activity"]["predicted_pIC50"]
            except HTTPException:row["target_activity"]=None
        valid.append(row)
    valid.sort(key=lambda r:(-r["screening_score"]["value"],r["toxicity"]["toxicity_probability"]))
    ranked=pareto_rank(valid,["activity","toxicity","solubility","qed"] if target else ["toxicity","solubility","qed"])
    for r in ranked[:20]:r["structure_svg_b64"]=structure_svg_b64(parse_mol(r["smiles"]))
    funnel={"submitted":len(raw),"invalid_structures":invalid,"unique_valid_structures":len(seen),"duplicates_removed":duplicates,"passed_lipinski":len(seen)-lipinski_fail,"pains_flagged":pains_flagged,"scored":len(ranked),"top_candidates":min(20,len(ranked))}
    return {"funnel":funnel,"candidate_count":len(ranked),"candidates":ranked[:20],"pareto_count":sum(bool(r.get("pareto_optimal")) for r in ranked),"all_candidates":ranked}

def screen_library(raw_smiles,target=None):return _screen_records([{"smiles":x} for x in raw_smiles],target)

def screen_records(records,target=None):return _screen_records(records,target)

def start_screen_job(records,target=None):
    job_id=f"screen_{uuid.uuid4().hex[:12]}"
    with _LOCK:_JOBS[job_id]={"job_id":job_id,"status":"queued","processed":0,"total":len(records)}
    def work():
        with _LOCK:_JOBS[job_id]["status"]="running"
        try:
            result=screen_records(records,target)
            with _LOCK:_JOBS[job_id].update({"status":"completed","result":result,"processed":len(records)})
        except Exception as exc:
            with _LOCK:_JOBS[job_id].update({"status":"failed","error":str(exc)})
    _EXECUTOR.submit(work); return _JOBS[job_id]

def get_screen_job(job_id):
    job=_JOBS.get(job_id)
    if not job:raise HTTPException(status_code=404,detail="Screening job not found.")
    return job
