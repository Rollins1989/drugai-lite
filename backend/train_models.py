
"""
DrugAI Lite — reproducible model training and evaluation.

v3 upgrades:
- Random and Bemis–Murcko scaffold splits
- Regression/classification metrics beyond a single score
- Scaffold leakage diagnostics
- Final deployed models are trained on the scaffold split
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

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
        return [
            Descriptors.MolWt(mol), Crippen.MolLogP(mol), Descriptors.TPSA(mol),
            Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol), Lipinski.NumAromaticRings(mol),
            Descriptors.RingCount(mol), Descriptors.FractionCSP3(mol),
            Descriptors.NumHeteroatoms(mol), QED.qed(mol),
            Descriptors.NumValenceElectrons(mol),
        ]
    except Exception:
        return None


def canonical_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def murcko_key(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else "__ACYCLIC__"


def build_feature_matrix(smiles_list):
    rows, keep_idx = [], []
    for i, smi in enumerate(smiles_list):
        d = compute_descriptors(smi)
        if d is not None:
            rows.append(d)
            keep_idx.append(i)
    return np.asarray(rows, dtype=float), keep_idx


def scaffold_split(smiles, test_size=0.2, random_state=42):
    """Split by Murcko scaffold so no scaffold appears in both train and test."""
    groups = {}
    for idx, smi in enumerate(smiles):
        key = murcko_key(smi)
        groups.setdefault(key, []).append(idx)

    rng = np.random.RandomState(random_state)
    scaffold_keys = list(groups)
    rng.shuffle(scaffold_keys)

    target_test = max(1, int(round(len(smiles) * test_size)))
    test_idx, train_idx = [], []
    for key in scaffold_keys:
        bucket = groups[key]
        if len(test_idx) < target_test:
            test_idx.extend(bucket)
        else:
            train_idx.extend(bucket)

    # Guard against pathological tiny datasets.
    if not train_idx or not test_idx:
        raise ValueError("Scaffold split produced an empty train/test partition.")
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def split_report(smiles, train_idx, test_idx):
    train_scaffolds = {murcko_key(smiles[i]) for i in train_idx}
    test_scaffolds = {murcko_key(smiles[i]) for i in test_idx}
    overlap = train_scaffolds & test_scaffolds
    return {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_scaffolds": int(len(train_scaffolds)),
        "n_test_scaffolds": int(len(test_scaffolds)),
        "scaffold_overlap": int(len(overlap)),
        "scaffold_leakage": bool(overlap),
    }


def fit_regressors(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    return rf, gb


def regression_metrics(y, p_rf, p_gb):
    p = (p_rf + p_gb) / 2
    return {
        "rf_r2": round(float(r2_score(y, p_rf)), 4),
        "gb_r2": round(float(r2_score(y, p_gb)), 4),
        "ensemble_r2": round(float(r2_score(y, p)), 4),
        "ensemble_rmse": round(float(mean_squared_error(y, p) ** 0.5), 4),
        "ensemble_mae": round(float(mean_absolute_error(y, p)), 4),
    }


def fit_classifiers(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(n_estimators=250, max_depth=3, random_state=42)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    return rf, gb


def classification_metrics(y, p_rf, p_gb):
    p = (p_rf + p_gb) / 2
    pred = (p >= 0.5).astype(int)
    return {
        "rf_auc": round(float(roc_auc_score(y, p_rf)), 4),
        "gb_auc": round(float(roc_auc_score(y, p_gb)), 4),
        "ensemble_auc": round(float(roc_auc_score(y, p)), 4),
        "ensemble_pr_auc": round(float(average_precision_score(y, p)), 4),
        "ensemble_accuracy": round(float(accuracy_score(y, pred)), 4),
        "ensemble_precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "ensemble_recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "ensemble_f1": round(float(f1_score(y, pred, zero_division=0)), 4),
    }


def evaluate_regression_splits(smiles, X, y):
    results = {}
    random_train, random_test = train_test_split(
        np.arange(len(smiles)), test_size=0.2, random_state=42
    )
    scaffold_train, scaffold_test = scaffold_split(smiles)

    for name, tr, te in [
        ("random_split", random_train, random_test),
        ("scaffold_split", scaffold_train, scaffold_test),
    ]:
        scaler = StandardScaler().fit(X[tr])
        rf, gb = fit_regressors(scaler.transform(X[tr]), y[tr])
        p_rf = rf.predict(scaler.transform(X[te]))
        p_gb = gb.predict(scaler.transform(X[te]))
        results[name] = {
            **regression_metrics(y[te], p_rf, p_gb),
            **split_report(smiles, tr, te),
        }

    # Persist the scaffold-trained model used by the API.
    scaler = StandardScaler().fit(X[scaffold_train])
    rf, gb = fit_regressors(scaler.transform(X[scaffold_train]), y[scaffold_train])
    joblib.dump(rf, MODELS / "solubility_rf.joblib")
    joblib.dump(gb, MODELS / "solubility_gb.joblib")
    joblib.dump(scaler, MODELS / "solubility_scaler.joblib")
    return results


def evaluate_classification_splits(smiles, X, y):
    results = {}
    random_train, random_test = train_test_split(
        np.arange(len(smiles)), test_size=0.2, random_state=42, stratify=y
    )
    scaffold_train, scaffold_test = scaffold_split(smiles)

    # A scaffold split is not stratified. Fail loudly if it contains one class.
    if len(np.unique(y[scaffold_test])) < 2 or len(np.unique(y[scaffold_train])) < 2:
        raise ValueError(
            "Scaffold split has only one class in train/test. "
            "Use a larger dataset or a different random seed."
        )

    for name, tr, te in [
        ("random_split", random_train, random_test),
        ("scaffold_split", scaffold_train, scaffold_test),
    ]:
        scaler = StandardScaler().fit(X[tr])
        rf, gb = fit_classifiers(scaler.transform(X[tr]), y[tr])
        p_rf = rf.predict_proba(scaler.transform(X[te]))[:, 1]
        p_gb = gb.predict_proba(scaler.transform(X[te]))[:, 1]
        results[name] = {
            **classification_metrics(y[te], p_rf, p_gb),
            **split_report(smiles, tr, te),
            "positive_rate_test": round(float(y[te].mean()), 4),
        }

    scaler = StandardScaler().fit(X[scaffold_train])
    rf, gb = fit_classifiers(scaler.transform(X[scaffold_train]), y[scaffold_train])
    joblib.dump(rf, MODELS / "toxicity_rf.joblib")
    joblib.dump(gb, MODELS / "toxicity_gb.joblib")
    joblib.dump(scaler, MODELS / "toxicity_scaler.joblib")
    return results


def train_solubility():
    print("=== Solubility: random vs scaffold evaluation ===")
    df = pd.read_csv(BASE / "data" / "delaney.csv")
    smiles_all = df["smiles"].tolist()
    y_all = df["measured log solubility in mols per litre"].astype(float).values
    X, keep = build_feature_matrix(smiles_all)
    smiles = [smiles_all[i] for i in keep]
    y = y_all[keep]
    results = evaluate_regression_splits(smiles, X, y)
    return {
        "task": "regression",
        "dataset": "ESOL (Delaney)",
        "features": DESCRIPTOR_NAMES,
        "deployment_split": "scaffold_split",
        "evaluation": results,
    }


def train_toxicity():
    print("=== Tox21 NR-AR: random vs scaffold evaluation ===")
    df = pd.read_csv(BASE / "data" / "tox21.csv")[["smiles", "NR-AR"]].dropna()
    smiles_all = df["smiles"].tolist()
    y_all = df["NR-AR"].astype(int).values
    X, keep = build_feature_matrix(smiles_all)
    smiles = [smiles_all[i] for i in keep]
    y = y_all[keep]
    results = evaluate_classification_splits(smiles, X, y)
    return {
        "task": "classification",
        "dataset": "Tox21 NR-AR",
        "assay": "NR-AR",
        "features": DESCRIPTOR_NAMES,
        "deployment_split": "scaffold_split",
        "evaluation": results,
    }


if __name__ == "__main__":
    sol = train_solubility()
    tox = train_toxicity()
    metrics = {"version": "3.0.0", "solubility": sol, "toxicity": tox}
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
