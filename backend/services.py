"""Application services coordinating chemistry and model layers."""
import csv
import io

import numpy as np
from fastapi import HTTPException
from rdkit import Chem

from chemistry import compute_descriptors, lipinski_verdict, molecular_identity, nearest_analogs, parse_mol, structural_alerts, structure_svg_b64, veber_verdict
from config import MAX_BATCH_MOLECULES
from model_service import explain, predict_solubility, predict_toxicity, screening_score


def domain_signal(analogs):
    if not analogs:
        return {"nearest_tanimoto": None, "status": "UNKNOWN", "note": "No reference molecules available."}
    sim = analogs[0]["tanimoto"]
    return {"nearest_tanimoto": sim, "status": "HIGH" if sim >= 0.7 else ("MODERATE" if sim >= 0.4 else "LOW"),
            "method": "Maximum Tanimoto similarity to the bundled ESOL/Tox21 reference library.",
            "note": "Chemical-space proximity signal, not a formal statistical applicability-domain model or guarantee of prediction reliability."}


def full_analysis(smiles: str) -> dict:
    mol = parse_mol(smiles)
    if mol is None: return {"valid": False, "smiles": smiles, "error": "Could not parse SMILES"}
    desc = compute_descriptors(mol)
    sol, tox = predict_solubility(desc), predict_toxicity(desc)
    lip, veber, alerts = lipinski_verdict(desc), veber_verdict(mol), structural_alerts(mol)
    identity, analogs = molecular_identity(mol), nearest_analogs(mol)
    domain = domain_signal(analogs)
    score = screening_score(sol, tox, desc["QED"], alerts["pass"], lip["pass"], veber["pass"], domain["status"])
    return {"valid": True, "smiles": smiles, "identity": identity, "descriptors": desc,
            "solubility": sol, "toxicity": tox, "lipinski": lip, "veber": veber,
            "structural_alerts": alerts, "applicability_domain": domain, "nearest_analogs": analogs,
            "explanation": explain(desc), "screening_score": score, "structure_svg_b64": structure_svg_b64(mol)}


def parse_uploaded_smiles(content: str):
    content = content.lstrip("\ufeff")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines: raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if lines[0].lower().startswith("smiles,") or "smiles" in [x.strip().lower() for x in lines[0].split(",")]:
        reader = csv.DictReader(io.StringIO(content))
        if not reader.fieldnames or "smiles" not in [h.lower() for h in reader.fieldnames]:
            raise HTTPException(status_code=400, detail="CSV must contain a 'smiles' column.")
        key = next(h for h in reader.fieldnames if h.lower() == "smiles")
        return [(row.get(key) or "").strip() for row in reader if (row.get(key) or "").strip()]
    return [line.split()[0] for line in lines]


def screen_library(raw_smiles):
    if len(raw_smiles) > MAX_BATCH_MOLECULES:
        raise HTTPException(status_code=413, detail=f"Maximum batch size is {MAX_BATCH_MOLECULES:,} molecules.")
    total = len(raw_smiles); valid_results = []; invalid_count = lipinski_fail = pains_fail = duplicate_count = 0; seen = set()
    for smi in raw_smiles:
        mol = parse_mol(smi)
        if mol is None: invalid_count += 1; continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in seen: duplicate_count += 1; continue
        seen.add(canonical)
        desc = compute_descriptors(mol); lip = lipinski_verdict(desc)
        if not lip["pass"]: lipinski_fail += 1; continue
        veber, alerts = veber_verdict(mol), structural_alerts(mol)
        if not alerts["pass"]: pains_fail += 1
        sol, tox = predict_solubility(desc), predict_toxicity(desc)
        analogs = nearest_analogs(mol, limit=1); domain = domain_signal(analogs)
        score = screening_score(sol, tox, desc["QED"], alerts["pass"], lip["pass"], veber["pass"], domain["status"])
        valid_results.append({"smiles": smi, "canonical_smiles": canonical, "descriptors": desc, "solubility": sol, "toxicity": tox, "veber": veber, "structural_alerts": alerts, "applicability_domain": domain, "screening_score": score})
    valid_results.sort(key=lambda r: (-r["screening_score"]["value"], r["toxicity"]["toxicity_probability"]))
    top = valid_results[:20]
    for result in top: result["structure_svg_b64"] = structure_svg_b64(parse_mol(result["smiles"]))
    funnel = {"submitted": total, "invalid_structures": invalid_count, "unique_valid_structures": len(seen), "duplicates_removed": duplicate_count,
              "passed_lipinski": total - invalid_count - duplicate_count - lipinski_fail, "pains_flagged": pains_fail, "scored": len(valid_results), "top_candidates": len(top)}
    return {"funnel": funnel, "candidates": top}
