"""Library analytics: multi-objective ranking, chemical space, clustering, and comparisons."""
from __future__ import annotations
import numpy as np
from rdkit import Chem, DataStructs
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from chemistry import fingerprint, compute_descriptors, parse_mol

OBJECTIVES = {
    "activity": "max",
    "toxicity": "min",
    "solubility": "max",
    "qed": "max",
    "mw": "min",
    "logp": "min",
}

def _values(row):
    toxicity = row.get("toxicity_probability")
    if toxicity is None:
        toxicity = row.get("toxicity", {}).get("toxicity_probability", 1.0) if isinstance(row.get("toxicity"), dict) else row.get("toxicity", 1.0)
    solubility = row.get("solubility")
    if isinstance(solubility, dict):
        solubility = solubility.get("log_solubility_mol_per_L", -99)
    else:
        solubility = row.get("log_solubility_mol_per_L", -99) if solubility is None else solubility
    descriptors = row.get("descriptors", {})
    activity = row.get("activity", row.get("predicted_pIC50"))
    qed = row.get("qed", descriptors.get("QED", 0.0))
    mw = row.get("mw", descriptors.get("MolWt", 9999))
    logp = row.get("logp", descriptors.get("LogP", 9999))
    return {
        "activity": float(activity) if activity is not None else None,
        "toxicity": float(toxicity),
        "solubility": float(solubility),
        "qed": float(qed),
        "mw": float(mw),
        "logp": float(logp),
    }

def pareto_front(rows: list[dict], objectives: list[str]) -> list[dict]:
    usable=[]
    for r in rows:
        vals=_values(r)
        if all(vals.get(o) is not None for o in objectives):
            usable.append((r,vals))
    front=[]
    for i,(candidate,a) in enumerate(usable):
        dominated=False
        for j,(other,b) in enumerate(usable):
            if i==j: continue
            better_or_equal=True
            strictly_better=False
            for o in objectives:
                direction=OBJECTIVES[o]
                if direction=="max":
                    if b[o] < a[o]: better_or_equal=False; break
                    if b[o] > a[o]: strictly_better=True
                else:
                    if b[o] > a[o]: better_or_equal=False; break
                    if b[o] < a[o]: strictly_better=True
            if better_or_equal and strictly_better:
                dominated=True; break
        if not dominated:
            front.append(candidate)
    return front

def pareto_rank(rows: list[dict], objectives: list[str]) -> list[dict]:
    front=pareto_front(rows,objectives)
    ids={id(x) for x in front}
    out=[]
    for r in rows:
        x=dict(r)
        x["pareto_optimal"]=id(r) in ids
        out.append(x)
    return out

def chemical_space(smiles: list[str], labels: list[str]|None=None, n_clusters: int=6) -> dict:
    valid=[]
    fps=[]
    original=[]
    for i,s in enumerate(smiles):
        m=parse_mol(s)
        if m is None: continue
        valid.append(i); original.append(s); fps.append(fingerprint(m))
    if not fps: return {"points":[],"pca_explained_variance":[],"clusters":0}
    X=np.asarray([[int(bit) for bit in fp.ToBitString()] for fp in fps],dtype=np.uint8)
    if len(fps)==1:
        point={"index":valid[0],"smiles":original[0],"x":0.0,"y":0.0,"cluster":0,"label":labels[valid[0]] if labels else None}
        return {"points":[point],"clusters":1,"pca_explained_variance":[1.0,0.0]}
    n_components=min(2,X.shape[0],X.shape[1])
    pca=PCA(n_components=n_components,random_state=42)
    raw_coords=pca.fit_transform(X)
    coords=np.column_stack([raw_coords[:,0],raw_coords[:,1] if n_components>1 else np.zeros(len(fps))])
    explained=list(pca.explained_variance_ratio_)
    if len(explained)<2: explained.append(0.0)
    k=max(1,min(n_clusters,len(fps)))
    if len(fps)>1 and k>1:
        clusterer=AgglomerativeClustering(n_clusters=k,metric="euclidean",linkage="ward")
        clusters=clusterer.fit_predict(coords)
    else: clusters=np.zeros(len(fps),dtype=int)
    points=[]
    for j,(idx,s) in enumerate(zip(valid,original)):
        points.append({"index":idx,"smiles":s,"x":round(float(coords[j,0]),4),"y":round(float(coords[j,1]),4),"cluster":int(clusters[j]),"label":labels[idx] if labels else None})
    return {"points":points,"clusters":int(k),"pca_explained_variance":[round(float(x),4) for x in explained[:2]]}

def pairwise_similarity(smiles_a:str, smiles_b:str)->dict:
    a,b=parse_mol(smiles_a),parse_mol(smiles_b)
    if a is None or b is None: raise ValueError("Both SMILES must be valid.")
    return {"tanimoto":round(float(DataStructs.TanimotoSimilarity(fingerprint(a),fingerprint(b))),4)}

def dataset_quality(rows:list[dict])->dict:
    total=len(rows); valid=[]; invalid=0; seen=set(); duplicates=0
    mw=[]; logp=[]; qed=[]; scaffolds=set()
    from rdkit.Chem.Scaffolds import MurckoScaffold
    for r in rows:
        s=str(r.get("smiles","")).strip()
        m=parse_mol(s)
        if m is None: invalid+=1; continue
        c=Chem.MolToSmiles(m,canonical=True)
        if c in seen: duplicates+=1
        seen.add(c); valid.append(c)
        d=compute_descriptors(m); mw.append(d["MolWt"]); logp.append(d["LogP"]); qed.append(d["QED"])
        sc=MurckoScaffold.GetScaffoldForMol(m)
        scaffolds.add(Chem.MolToSmiles(sc,canonical=True) if sc.GetNumAtoms() else "__ACYCLIC__")
    return {
        "rows":total,"valid_structures":len(valid),"invalid_structures":invalid,
        "duplicate_structures":duplicates,"unique_molecules":len(seen),"unique_scaffolds":len(scaffolds),
        "properties":{"median_mw":round(float(np.median(mw)),2) if mw else None,"median_logp":round(float(np.median(logp)),2) if logp else None,"median_qed":round(float(np.median(qed)),3) if qed else None},
    }
