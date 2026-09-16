"""Train a target-specific activity model from ChEMBL.

Default target: EGFR / CHEMBL203
Endpoint: IC50, represented as pIC50 using ChEMBL pChEMBL values.

Pipeline:
    ChEMBL -> clean/aggregate -> Morgan fingerprints -> scaffold split
    -> Random Forest + Gradient Boosting -> pIC50 prediction.

The script stores retrieval/query metadata and a simple mean baseline so
future runs are reproducible and model performance is not presented without
context.

Run from backend:
    python train_target_activity.py --target CHEMBL203 --name egfr
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from rdkit import Chem
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


def fetch_chembl(target_id, max_records=5000, request_timeout=60):
    """Fetch usable human IC50 records and return data plus query metadata."""
    rows = []
    offset = 0
    page_size = min(1000, max_records)
    query = {
        "target_chembl_id": target_id,
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_units": "nM",
        "target_organism": "Homo sapiens",
        "max_records": max_records,
    }
    retrieved_at = datetime.now(timezone.utc).isoformat()

    while len(rows) < max_records:
        params = {
            "target_chembl_id": target_id,
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_units": "nM",
            "limit": page_size,
            "offset": offset,
        }
        response = requests.get(API, params=params, timeout=request_timeout)
        response.raise_for_status()
        payload = response.json()
        activities = payload.get("activities", [])
        if not activities:
            break

        for activity in activities:
            if (
                activity.get("canonical_smiles")
                and activity.get("standard_value") is not None
                and activity.get("pchembl_value") is not None
                and activity.get("target_organism") == "Homo sapiens"
                and not activity.get("data_validity_comment")
            ):
                rows.append({
                    "molecule_chembl_id": activity.get("molecule_chembl_id"),
                    "canonical_smiles": activity["canonical_smiles"],
                    "standard_value_nM": float(activity["standard_value"]),
                    "pchembl_value": float(activity["pchembl_value"]),
                    "assay_chembl_id": activity.get("assay_chembl_id"),
                    "document_chembl_id": activity.get("document_chembl_id"),
                })

        offset += len(activities)
        if len(activities) < page_size:
            break
        time.sleep(0.15)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("ChEMBL returned no usable IC50 records for this target.")

    df = (
        df.groupby("canonical_smiles", as_index=False)
        .agg(
            molecule_chembl_id=("molecule_chembl_id", "first"),
            standard_value_nM=("standard_value_nM", "median"),
            pchembl_value=("pchembl_value", "median"),
            n_measurements=("pchembl_value", "size"),
        )
    )
    metadata = {
        "source": "ChEMBL activity API",
        "api": API,
        "retrieved_at_utc": retrieved_at,
        "query": query,
        "filters_applied": [
            "canonical_smiles present",
            "standard_value present",
            "pchembl_value present",
            "target_organism == Homo sapiens",
            "data_validity_comment absent",
            "duplicate canonical SMILES aggregated by median",
        ],
        "aggregation": "median pIC50 and median standard_value_nM per canonical SMILES",
    }
    return df, metadata


def murcko(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES during scaffold calculation: {smiles}")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else "__ACYCLIC__"


def scaffold_split(smiles, test_size=0.2, seed=42):
    """Create a scaffold-disjoint split while keeping test size near the target."""
    groups = {}
    for index, smi in enumerate(smiles):
        groups.setdefault(murcko(smi), []).append(index)

    rng = np.random.RandomState(seed)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda key: len(groups[key]), reverse=True)
    target_test = max(1, round(len(smiles) * test_size))
    test, train = [], []
    for key in keys:
        group = groups[key]
        current = len(test)
        if abs((current + len(group)) - target_test) < abs(current - target_test):
            test.extend(group)
        else:
            train.extend(group)

    if not test and train:
        key = min(groups, key=lambda k: len(groups[k]))
        moved = set(groups[key])
        test.extend(groups[key])
        train = [i for i in train if i not in moved]
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
        fps.append(FP_GENERATOR.GetFingerprintAsNumPy(mol))
        valid.append(i)
    if not fps:
        return np.empty((0, 2048), dtype=np.uint8), valid
    return np.asarray(fps, dtype=np.uint8), valid


def train(df, test_size=0.2, seed=42):
    smiles_all = df["canonical_smiles"].tolist()
    X, keep = fingerprints(smiles_all)
    df = df.iloc[keep].reset_index(drop=True)
    y = df["pchembl_value"].astype(float).values
    smiles = df["canonical_smiles"].tolist()

    tr, te = scaffold_split(smiles, test_size=test_size, seed=seed)
    scaler = StandardScaler(with_mean=False).fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])

    rf = RandomForestRegressor(n_estimators=400, max_depth=18, random_state=seed, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=seed)
    rf.fit(Xtr, y[tr])
    gb.fit(Xtr, y[tr])

    p_rf = rf.predict(Xte)
    p_gb = gb.predict(Xte)
    p = (p_rf + p_gb) / 2
    baseline = np.full(len(te), float(np.mean(y[tr])))

    train_scaffolds = {murcko(smiles[i]) for i in tr}
    test_scaffolds = {murcko(smiles[i]) for i in te}
    metrics = {
        "n_molecules": int(len(df)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "test_fraction_actual": round(float(len(te) / len(df)), 4),
        "n_train_scaffolds": len(train_scaffolds),
        "n_test_scaffolds": len(test_scaffolds),
        "scaffold_overlap": len(train_scaffolds & test_scaffolds),
        "rf_r2": round(float(r2_score(y[te], p_rf)), 4),
        "gb_r2": round(float(r2_score(y[te], p_gb)), 4),
        "ensemble_r2": round(float(r2_score(y[te], p)), 4),
        "ensemble_rmse": round(float(mean_squared_error(y[te], p) ** 0.5), 4),
        "ensemble_mae": round(float(mean_absolute_error(y[te], p)), 4),
        "mean_baseline_r2": round(float(r2_score(y[te], baseline)), 4),
        "mean_baseline_rmse": round(float(mean_squared_error(y[te], baseline) ** 0.5), 4),
        "mean_baseline_mae": round(float(mean_absolute_error(y[te], baseline)), 4),
        "target_type": "IC50",
        "target_label": "pIC50",
        "split": "Bemis-Murcko scaffold",
        "split_seed": seed,
        "test_size_requested": test_size,
        "fingerprint": "Morgan radius=2, 2048 bits",
    }
    return df, scaler, rf, gb, metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="CHEMBL203")
    parser.add_argument("--name", default="egfr")
    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.test_size < 0.5:
        raise SystemExit("--test-size must be between 0 and 0.5")

    print(f"Downloading ChEMBL IC50 data for {args.target}...")
    df, retrieval_metadata = fetch_chembl(args.target, args.max_records)
    dataset_path = TARGET_DATA / f"{args.name}_ic50.csv"
    df.to_csv(dataset_path, index=False)
    retrieval_metadata["dataset_file"] = str(dataset_path.relative_to(BASE))
    retrieval_metadata["dataset_sha256"] = sha256_file(dataset_path)

    df, scaler, rf, gb, metrics = train(df, test_size=args.test_size, seed=args.seed)
    out = TARGET_MODELS / args.name
    out.mkdir(exist_ok=True)
    joblib.dump(scaler, out / "scaler.joblib")
    joblib.dump(rf, out / "rf.joblib")
    joblib.dump(gb, out / "gb.joblib")

    metadata = {
        "target_chembl_id": args.target,
        "model": metrics,
        "data": retrieval_metadata,
        "training": {
            "random_seed": args.seed,
            "test_size_requested": args.test_size,
            "algorithm": "Random Forest + Gradient Boosting ensemble",
            "fingerprint": "Morgan radius=2, 2048 bits",
            "feature_scaling": "StandardScaler(with_mean=False); retained for compatibility with bundled artifacts",
        },
    }
    (out / "metrics.json").write_text(json.dumps(metadata, indent=2))
    (out / "dataset_metadata.json").write_text(json.dumps(retrieval_metadata, indent=2))

    print(json.dumps(metadata, indent=2))
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
