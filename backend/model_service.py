"""Model loading, prediction, uncertainty diagnostics, and explainability."""
from __future__ import annotations
import json
import joblib
import numpy as np
from fastapi import HTTPException
from chemistry import MORGAN, descriptor_vector, parse_mol
from config import MODELS_DIR, DESCRIPTOR_NAMES

def _load(path):
    if not path.exists(): raise RuntimeError(f"Required model artifact is missing: {path}")
    return joblib.load(path)
def _load_json(path, default=None):
    if not path.exists(): return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))

sol_rf=_load(MODELS_DIR/"solubility_rf.joblib"); sol_gb=_load(MODELS_DIR/"solubility_gb.joblib"); sol_scaler=_load(MODELS_DIR/"solubility_scaler.joblib")
tox_rf=_load(MODELS_DIR/"toxicity_rf.joblib"); tox_gb=_load(MODELS_DIR/"toxicity_gb.joblib"); tox_scaler=_load(MODELS_DIR/"toxicity_scaler.joblib")
METRICS=_load_json(MODELS_DIR/"metrics.json"); MODEL_VERSION=_load_json(MODELS_DIR/"model_version.json",{"version":"unknown"})
TARGET_MODELS={}
TARGET_DIR=MODELS_DIR/"targets"
if TARGET_DIR.exists():
    for target_dir in TARGET_DIR.iterdir():
        if target_dir.is_dir():
            req=[target_dir/"rf.joblib",target_dir/"gb.joblib",target_dir/"scaler.joblib"]
            if all(p.exists() for p in req):
                TARGET_MODELS[target_dir.name]={"rf":_load(req[0]),"gb":_load(req[1]),"scaler":_load(req[2]),"metrics":_load_json(target_dir/"metrics.json")}

def agreement_signal(disagreement,thresholds):
    low,moderate=thresholds
    level="HIGH_AGREEMENT" if disagreement<low else ("MODERATE_AGREEMENT" if disagreement<moderate else "LOW_AGREEMENT")
    return {"absolute_disagreement":round(float(disagreement),4),"level":level,"method":"Absolute RF/Gradient Boosting prediction difference.","interpretation":"Ensemble agreement is a diagnostic signal, not calibrated probability of correctness."}

def _rf_tree_spread(model,X):
    est=getattr(model,"estimators_",[])
    if not est:return None
    values=np.asarray([float(tree.predict(X)[0]) for tree in est])
    return {"q10":round(float(np.quantile(values,.10)),4),"q90":round(float(np.quantile(values,.90)),4),"std":round(float(np.std(values)),4),"n_models":len(values)}

def predict_solubility(desc):
    X=sol_scaler.transform(descriptor_vector(desc)); rf=float(sol_rf.predict(X)[0]); gb=float(sol_gb.predict(X)[0]); mean=(rf+gb)/2
    s=_rf_tree_spread(sol_rf,X); interval=[round(s["q10"],2),round(s["q90"],2)] if s else None
    return {"log_solubility_mol_per_L":round(mean,2),"label":"GOOD" if mean>-2 else ("MODERATE" if mean>-4 else "POOR"),"model_agreement":agreement_signal(abs(rf-gb),(.4,1.0)),"rf_prediction":round(rf,2),"gb_prediction":round(gb,2),"empirical_ensemble_interval_80pct":interval,"uncertainty_note":"Empirical RF-tree spread; not a calibrated prediction interval.","label_definition":"GOOD > -2; MODERATE -4 to -2; POOR < -4."}

def predict_toxicity(desc):
    X=tox_scaler.transform(descriptor_vector(desc)); rf=float(tox_rf.predict_proba(X)[0][1]); gb=float(tox_gb.predict_proba(X)[0][1]); mean=(rf+gb)/2
    s=_rf_tree_spread(tox_rf,X); interval=[round(s["q10"],3),round(s["q90"],3)] if s else None
    return {"toxicity_probability":round(mean,3),"label":"LOW" if mean<.3 else ("MODERATE" if mean<.6 else "HIGH"),"model_agreement":agreement_signal(abs(rf-gb),(.1,.25)),"assay":METRICS.get("toxicity",{}).get("assay","Tox21 NR-AR")+" (androgen receptor nuclear signalling)","rf_prediction":round(rf,3),"gb_prediction":round(gb,3),"empirical_ensemble_interval_80pct":interval,"uncertainty_note":"Empirical RF-tree probability spread; not a calibrated confidence interval.","label_definition":"LOW < 0.30; MODERATE 0.30–0.60; HIGH >= 0.60","note":"Probability refers to the modeled Tox21 NR-AR assay endpoint, not general human toxicity."}

def explain(desc):
    def ranked(model):
        vals=getattr(model,"feature_importances_",np.zeros(len(DESCRIPTOR_NAMES)))
        return [{"descriptor":n,"value":desc[n],"importance":round(float(i),4)} for n,i in sorted(zip(DESCRIPTOR_NAMES,vals),key=lambda x:-x[1])[:6]]
    return {"toxicity":ranked(tox_rf),"solubility":ranked(sol_rf),"method":"Global tree feature_importances_; per-molecule local sensitivity is reported separately."}

def local_sensitivity(desc):
    """Estimate directional one-feature sensitivity around the training distribution.

    The deployed models require scaled descriptor inputs. The previous implementation
    accidentally predicted on raw descriptors and used the molecule's own values as
    its "median", making the diagnostic effectively zero.
    """
    base=descriptor_vector(desc)
    sol_X=sol_scaler.transform(base)
    tox_X=tox_scaler.transform(base)
    sol_base=float((sol_rf.predict(sol_X)[0]+sol_gb.predict(sol_X)[0])/2)
    tox_base=float((tox_rf.predict_proba(tox_X)[0][1]+tox_gb.predict_proba(tox_X)[0][1])/2)
    sol_mean=np.asarray(getattr(sol_scaler,"mean_",base[0]),dtype=float)
    sol_scale=np.maximum(np.asarray(getattr(sol_scaler,"scale_",np.ones(len(DESCRIPTOR_NAMES))),dtype=float),1e-9)
    tox_mean=np.asarray(getattr(tox_scaler,"mean_",base[0]),dtype=float)
    tox_scale=np.maximum(np.asarray(getattr(tox_scaler,"scale_",np.ones(len(DESCRIPTOR_NAMES))),dtype=float),1e-9)
    rows=[]
    for i,name in enumerate(DESCRIPTOR_NAMES):
        sol_plus=base.copy(); sol_plus[0,i]=sol_mean[i]+sol_scale[i]
        sol_minus=base.copy(); sol_minus[0,i]=sol_mean[i]-sol_scale[i]
        tox_plus=base.copy(); tox_plus[0,i]=tox_mean[i]+tox_scale[i]
        tox_minus=base.copy(); tox_minus[0,i]=tox_mean[i]-tox_scale[i]
        sol_p=float((sol_rf.predict(sol_scaler.transform(sol_plus))[0]+sol_gb.predict(sol_scaler.transform(sol_plus))[0])/2)
        sol_m=float((sol_rf.predict(sol_scaler.transform(sol_minus))[0]+sol_gb.predict(sol_scaler.transform(sol_minus))[0])/2)
        tox_p=float((tox_rf.predict_proba(tox_scaler.transform(tox_plus))[0][1]+tox_gb.predict_proba(tox_scaler.transform(tox_plus))[0][1])/2)
        tox_m=float((tox_rf.predict_proba(tox_scaler.transform(tox_minus))[0][1]+tox_gb.predict_proba(tox_scaler.transform(tox_minus))[0][1])/2)
        rows.append({"descriptor":name,
                     "solubility_delta":round(sol_p-sol_base,4),
                     "solubility_delta_minus_1sd":round(sol_m-sol_base,4),
                     "solubility_delta_plus_1sd":round(sol_p-sol_base,4),
                     "toxicity_probability_delta":round(tox_p-tox_base,4),
                     "toxicity_probability_delta_minus_1sd":round(tox_m-tox_base,4),
                     "toxicity_probability_delta_plus_1sd":round(tox_p-tox_base,4)})
    return {"method":"One-feature ±1 training-standard-deviation perturbation; directional diagnostic, not causal attribution.","baseline":{"solubility":round(sol_base,4),"toxicity_probability":round(tox_base,4)},"features":rows}

def screening_score(sol,tox,qed,pains_pass,lipinski_pass,veber_pass,domain_status):
    sol_component={"GOOD":1.0,"MODERATE":.6,"POOR":.2}[sol["label"]]; domain_component={"HIGH":1.0,"MODERATE":.8,"LOW":.5,"UNKNOWN":.6}[domain_status]
    alert_component=1.0 if pains_pass else .55; rule_component=(1 if lipinski_pass else .7)*(1 if veber_pass else .85)
    components={"toxicity":.28*(1-tox["toxicity_probability"]),"solubility":.25*sol_component,"qed":.17*qed,"chemical_space_proximity":.15*domain_component,"pains":.10*alert_component,"drug_likeness_rules":.05*rule_component}
    return {"value":round(float(np.clip(sum(components.values()),0,1)),3),"components":{k:round(float(v),4) for k,v in components.items()},"weights":{"toxicity":.28,"solubility":.25,"qed":.17,"chemical_space_proximity":.15,"pains":.10,"drug_likeness_rules":.05},"type":"heuristic","note":"Hand-weighted candidate-prioritization heuristic; not a validated efficacy, safety, developability, or clinical score."}

def target_activity_prediction(smiles,target_name):
    bundle=TARGET_MODELS.get(target_name)
    if bundle is None: raise HTTPException(status_code=404,detail=f"No trained target model named '{target_name}'. Available: {sorted(TARGET_MODELS)}.")
    mol=parse_mol(smiles)
    if mol is None: raise HTTPException(status_code=400,detail="Could not parse SMILES.")
    vector=np.asarray(MORGAN.GetFingerprintAsNumPy(mol),dtype=np.uint8).reshape(1,-1); X=bundle["scaler"].transform(vector)
    rf=float(bundle["rf"].predict(X)[0]); gb=float(bundle["gb"].predict(X)[0]); pred=(rf+gb)/2; s=_rf_tree_spread(bundle["rf"],X)
    interval=[round(s["q10"],3),round(s["q90"],3)] if s else None
    return {"target":target_name,"predicted_pIC50":round(pred,3),"predicted_IC50_nM":round(float(10**(9-pred)),2),"rf_pIC50":round(rf,3),"gb_pIC50":round(gb,3),"model_agreement":agreement_signal(abs(rf-gb),(.35,.75)),"empirical_ensemble_interval_80pct":interval,"uncertainty_note":"Empirical RF-tree spread; not a calibrated prediction interval.","pIC50_threshold_6_exceeded":bool(pred>=6),"training_evaluation":bundle["metrics"],"note":"pIC50 is a model prediction, not experimental activity. The pIC50=6 flag is a screening threshold, not an efficacy claim."}
