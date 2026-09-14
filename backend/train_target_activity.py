
"""
Build and train a target-specific activity model from ChEMBL.

Default target:
    EGFR / CHEMBL203
Default assay endpoint:
    IC50, converted to pIC50 by ChEMBL's pChEMBL value.

Pipeline:
    ChEMBL -> clean/aggregate -> Morgan fingerprints -> scaffold split
    -> Random Forest + Gradient Boosting -> pIC50 prediction.

Run from backend:
    python train_target_activity.py --target CHEMBL203 --name egfr
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
TARGET_DATA = BASE / "data" / "targets"
TARGET_MODELS = BASE / "models" / "targets"
TARGET_DATA.mkdir(parents=True, exist_ok=True)
TARGET_MODELS.mkdir(parents=True, exist_ok=True)

API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fetch_chembl(target_id, max_records=5000):
    rows = []
    offset = 0
    page_size = min(1000, max_records)

    while len(rows) < max_records:
        params = {
            "target_chembl_id": target_id,
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_units": "nM",
            "limit": page_size,
            "offset": offset,
        }
        response = requests.get(API, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        activities = payload.get("activities", [])
        if not activities:
            break

        for a in activities:
            if (
                a.get("canonical_smiles")
                and a.get("standard_value") is not None
                and a.get("pchembl_value") is not None
                and a.get("target_organism") == "Homo sapiens"
                and not a.get("data_validity_comment")
            ):
                rows.append({
                    "molecule_chembl_id": a.get("molecule_chembl_id"),
                    "canonical_smiles": a["canonical_smiles"],
                    "standard_value_nM": float(a["standard_value"]),
                    "pchembl_value": float(a["pchembl_value"]),
                    "assay_chembl_id": a.get("assay_chembl_id"),
                    "document_chembl_id": a.get("document_chembl_id"),
                })

        offset += len(activities)
        if len(activities) < page_size:
            break
        time.sleep(0.15)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("ChEMBL returned no usable IC50 records for this target.")

    # Multiple assays can report the same molecule. Median aggregation gives
    # one training label per chemical and avoids duplicated compounds leaking
    # across train/test.
    df = (
        df.groupby("canonical_smiles", as_index=False)
        .agg(
            molecule_chembl_id=("molecule_chembl_id", "first"),
            standard_value_nM=("standard_value_nM", "median"),
            pchembl_value=("pchembl_value", "median"),
            n_measurements=("pchembl_value", "size"),
        )
    )
    return df


def murcko(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else "__ACYCLIC__"


def scaffold_split(smiles, test_size=0.2, seed=42):
    groups = {}
    for i, smi in enumerate(smiles):
        groups.setdefault(murcko(smi), []).append(i)
    rng = np.random.RandomState(seed)
    keys = list(groups)
    rng.shuffle(keys)
    target_test = max(1, round(len(smiles) * test_size))
    test, train = [], []
    for key in keys:
        if len(test) < target_test:
            test.extend(groups[key])
        else:
            train.extend(groups[key])
    if not train or not test:
        raise RuntimeError("Could not construct a non-empty scaffold split.")
    return np.array(sorted(train)), np.array(sorted(test))


def fingerprints(smiles):
    fps = []
    valid = []

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            continue

        fp = FP_GENERATOR.GetFingerprintAsNumPy(mol)
        fps.append(fp)
        valid.append(i)

    if not fps:
        return np.empty((0, 2048), dtype=np.uint8), valid

    X = np.asarray(fps, dtype=np.uint8)

    return X, valid


def train(df):
    smiles_all = df["canonical_smiles"].tolist()
    X, keep = fingerprints(smiles_all)
    df = df.iloc[keep].reset_index(drop=True)
    y = df["pchembl_value"].astype(float).values
    smiles = df["canonical_smiles"].tolist()

    tr, te = scaffold_split(smiles)
    scaler = StandardScaler(with_mean=False).fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])

    rf = RandomForestRegressor(
        n_estimators=400, max_depth=18, random_state=42, n_jobs=-1
    )
    gb = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, random_state=42
    )
    rf.fit(Xtr, y[tr])
    gb.fit(Xtr, y[tr])

    p_rf = rf.predict(Xte)
    p_gb = gb.predict(Xte)
    p = (p_rf + p_gb) / 2

    train_scaffolds = {murcko(smiles[i]) for i in tr}
    test_scaffolds = {murcko(smiles[i]) for i in te}
    metrics = {
        "n_molecules": int(len(df)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_train_scaffolds": len(train_scaffolds),
        "n_test_scaffolds": len(test_scaffolds),
        "scaffold_overlap": len(train_scaffolds & test_scaffolds),
        "rf_r2": round(float(r2_score(y[te], p_rf)), 4),
        "gb_r2": round(float(r2_score(y[te], p_gb)), 4),
        "ensemble_r2": round(float(r2_score(y[te], p)), 4),
        "ensemble_rmse": round(float(mean_squared_error(y[te], p) ** 0.5), 4),
        "ensemble_mae": round(float(mean_absolute_error(y[te], p)), 4),
        "target_type": "IC50",
        "target_label": "pIC50",
        "split": "Bemis-Murcko scaffold",
    }

    return df, scaler, rf, gb, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="CHEMBL203")
    parser.add_argument("--name", default="egfr")
    parser.add_argument("--max-records", type=int, default=5000)
    args = parser.parse_args()

    print(f"Downloading ChEMBL IC50 data for {args.target}...")
    df = fetch_chembl(args.target, args.max_records)
    df.to_csv(TARGET_DATA / f"{args.name}_ic50.csv", index=False)

    df, scaler, rf, gb, metrics = train(df)
    out = TARGET_MODELS / args.name
    out.mkdir(exist_ok=True)
    joblib.dump(scaler, out / "scaler.joblib")
    joblib.dump(rf, out / "rf.joblib")
    joblib.dump(gb, out / "gb.joblib")
    (out / "metrics.json").write_text(json.dumps({
        "target_chembl_id": args.target,
        "model": metrics,
    }, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
