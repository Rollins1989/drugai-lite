"""
DrugAI Lite — model training.

Trains two real, evaluated models on real public datasets:
  1. Solubility regressor  — ESOL (Delaney) dataset, 1128 molecules
  2. Toxicity classifier   — Tox21 NR-AR assay, ~7800 molecules (binary, active/inactive)

Both use RDKit 2D descriptors as features and a small ensemble
(RandomForest + GradientBoosting) so we get real model-disagreement
based uncertainty at inference time, not a fabricated confidence score.

Run: python train_models.py
Outputs: backend/models/*.joblib + backend/models/metrics.json
"""
import json
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, Lipinski
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

DESCRIPTOR_NAMES = [
    "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "RingCount",
    "FractionCSP3", "NumHeteroatoms", "QED", "NumValenceElectrons",
]


def compute_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        vals = [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Descriptors.TPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Lipinski.NumAromaticRings(mol),
            Descriptors.RingCount(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.NumHeteroatoms(mol),
            QED.qed(mol),
            Descriptors.NumValenceElectrons(mol),
        ]
        return vals
    except Exception:
        return None


def build_feature_matrix(smiles_list):
    rows, keep_idx = [], []
    for i, smi in enumerate(smiles_list):
        d = compute_descriptors(smi)
        if d is not None:
            rows.append(d)
            keep_idx.append(i)
    return np.array(rows), keep_idx


def train_solubility():
    print("=== Training solubility model (ESOL / Delaney) ===")
    df = pd.read_csv("data/delaney.csv")
    smiles = df["smiles"].tolist()
    y = df["measured log solubility in mols per litre"].values

    X, keep = build_feature_matrix(smiles)
    y = y[keep]
    print(f"Usable molecules: {len(keep)} / {len(smiles)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42)
    rf.fit(X_train_s, y_train)
    gb.fit(X_train_s, y_train)

    preds_rf = rf.predict(X_test_s)
    preds_gb = gb.predict(X_test_s)
    ensemble_pred = (preds_rf + preds_gb) / 2

    metrics = {
        "rf_r2": round(r2_score(y_test, preds_rf), 4),
        "gb_r2": round(r2_score(y_test, preds_gb), 4),
        "ensemble_r2": round(r2_score(y_test, ensemble_pred), 4),
        "ensemble_rmse": round(mean_squared_error(y_test, ensemble_pred) ** 0.5, 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "dataset": "ESOL (Delaney) — 1128 molecules, MoleculeNet",
    }
    print(json.dumps(metrics, indent=2))

    joblib.dump(rf, "models/solubility_rf.joblib")
    joblib.dump(gb, "models/solubility_gb.joblib")
    joblib.dump(scaler, "models/solubility_scaler.joblib")
    return metrics


def train_toxicity():
    print("\n=== Training toxicity model (Tox21 / NR-AR assay) ===")
    df = pd.read_csv("data/tox21.csv")
    task = "NR-AR"
    df = df[["smiles", task]].dropna(subset=[task])
    smiles = df["smiles"].tolist()
    y = df[task].astype(int).values
    print(f"Assay: {task}  Positives: {y.sum()}  Negatives: {(y==0).sum()}")

    X, keep = build_feature_matrix(smiles)
    y = y[keep]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(n_estimators=250, max_depth=3, random_state=42)
    rf.fit(X_train_s, y_train)
    gb.fit(X_train_s, y_train)

    proba_rf = rf.predict_proba(X_test_s)[:, 1]
    proba_gb = gb.predict_proba(X_test_s)[:, 1]
    ensemble_proba = (proba_rf + proba_gb) / 2
    ensemble_pred = (ensemble_proba > 0.5).astype(int)

    metrics = {
        "rf_auc": round(roc_auc_score(y_test, proba_rf), 4),
        "gb_auc": round(roc_auc_score(y_test, proba_gb), 4),
        "ensemble_auc": round(roc_auc_score(y_test, ensemble_proba), 4),
        "ensemble_accuracy": round(accuracy_score(y_test, ensemble_pred), 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "assay": task,
        "dataset": "Tox21 (NR-AR nuclear receptor assay), MoleculeNet",
    }
    print(json.dumps(metrics, indent=2))

    joblib.dump(rf, "models/toxicity_rf.joblib")
    joblib.dump(gb, "models/toxicity_gb.joblib")
    joblib.dump(scaler, "models/toxicity_scaler.joblib")
    return metrics


if __name__ == "__main__":
    sol_metrics = train_solubility()
    tox_metrics = train_toxicity()
    with open("models/metrics.json", "w") as f:
        json.dump({"solubility": sol_metrics, "toxicity": tox_metrics}, f, indent=2)
    print("\nSaved models + metrics.json")
