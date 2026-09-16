"""
DrugAI Lite — reproducible model training and evaluation.

Phase 3 upgrades:
- Random and Bemis–Murcko scaffold splits
- Regression/classification metrics plus simple baselines
- ROC, PR and calibration diagnostics for classification
- Actual-vs-predicted and residual diagnostics for regression
- Scaffold leakage diagnostics
- Reproducible run metadata
- Final deployed models trained on the scaffold split

Run from backend:
    python train_models.py

Evaluation artifacts are written to backend/models/evaluation/.
"""
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, mean_absolute_error, mean_squared_error,
    precision_score, precision_recall_curve, r2_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
MODELS = BASE / "models"
EVALUATION = MODELS / "evaluation"
MODELS.mkdir(exist_ok=True)
EVALUATION.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
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


def scaffold_split(smiles, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split by Murcko scaffold with deterministic group-disjoint partitions."""
    groups = {}
    for idx, smi in enumerate(smiles):
        groups.setdefault(murcko_key(smi), []).append(idx)

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
        "actual_test_fraction": round(float(len(test_idx) / (len(train_idx) + len(test_idx))), 4),
        "n_train_scaffolds": int(len(train_scaffolds)),
        "n_test_scaffolds": int(len(test_scaffolds)),
        "scaffold_overlap": int(len(overlap)),
        "scaffold_leakage": bool(overlap),
    }


def fit_regressors(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    return rf, gb


def regression_metrics(y, p_rf, p_gb, y_train):
    p = (p_rf + p_gb) / 2
    baseline = np.full_like(y, float(np.mean(y_train)), dtype=float)
    return {
        "rf_r2": round(float(r2_score(y, p_rf)), 4),
        "gb_r2": round(float(r2_score(y, p_gb)), 4),
        "ensemble_r2": round(float(r2_score(y, p)), 4),
        "ensemble_rmse": round(float(mean_squared_error(y, p) ** 0.5), 4),
        "ensemble_mae": round(float(mean_absolute_error(y, p)), 4),
        "baseline_mean_r2": round(float(r2_score(y, baseline)), 4),
        "baseline_mean_rmse": round(float(mean_squared_error(y, baseline) ** 0.5), 4),
        "baseline_mean_mae": round(float(mean_absolute_error(y, baseline)), 4),
    }


def fit_classifiers(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    gb = GradientBoostingClassifier(n_estimators=250, max_depth=3, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    return rf, gb


def classification_metrics(y, p_rf, p_gb, y_train):
    p = (p_rf + p_gb) / 2
    pred = (p >= 0.5).astype(int)
    baseline_prob = float(np.mean(y_train))
    baseline = np.full(len(y), baseline_prob)
    baseline_pred = np.full(len(y), int(baseline_prob >= 0.5))
    return {
        "rf_auc": round(float(roc_auc_score(y, p_rf)), 4),
        "gb_auc": round(float(roc_auc_score(y, p_gb)), 4),
        "ensemble_auc": round(float(roc_auc_score(y, p)), 4),
        "ensemble_pr_auc": round(float(average_precision_score(y, p)), 4),
        "ensemble_accuracy": round(float(accuracy_score(y, pred)), 4),
        "ensemble_precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "ensemble_recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "ensemble_f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "ensemble_brier": round(float(brier_score_loss(y, p)), 4),
        "baseline_prevalence": round(baseline_prob, 4),
        "baseline_accuracy": round(float(accuracy_score(y, baseline_pred)), 4),
        "baseline_pr_auc": round(float(average_precision_score(y, baseline)), 4),
        "baseline_brier": round(float(brier_score_loss(y, baseline)), 4),
    }


def save_regression_plots(y, prediction, residual, prefix):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y, prediction, alpha=0.55)
    lo, hi = min(y.min(), prediction.min()), max(y.max(), prediction.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{prefix}: observed vs predicted")
    fig.tight_layout()
    fig.savefig(EVALUATION / f"{prefix}_observed_vs_predicted.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(residual, bins=25)
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Residual (observed − predicted)")
    ax.set_ylabel("Count")
    ax.set_title(f"{prefix}: residual distribution")
    fig.tight_layout()
    fig.savefig(EVALUATION / f"{prefix}_residuals.png", dpi=160)
    plt.close(fig)


def save_classification_plots(y, probability, prefix):
    fpr, tpr, _ = roc_curve(y, probability)
    precision, recall, _ = precision_recall_curve(y, probability)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y, probability):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{prefix}: ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(EVALUATION / f"{prefix}_roc.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {average_precision_score(y, probability):.3f}")
    ax.axhline(float(np.mean(y)), linestyle="--", label="Prevalence baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{prefix}: precision-recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(EVALUATION / f"{prefix}_pr.png", dpi=160)
    plt.close(fig)

    # Reliability diagram using quantile bins; avoids assuming uniformly populated bins.
    order = np.argsort(probability)
    bins = np.array_split(order, min(10, len(order)))
    mean_pred, frac_pos = [], []
    for b in bins:
        if len(b):
            mean_pred.append(float(np.mean(probability[b])))
            frac_pos.append(float(np.mean(y[b])))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title(f"{prefix}: reliability diagram")
    fig.tight_layout()
    fig.savefig(EVALUATION / f"{prefix}_calibration.png", dpi=160)
    plt.close(fig)


def evaluate_regression_splits(smiles, X, y):
    results = {}
    random_train, random_test = train_test_split(
        np.arange(len(smiles)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    scaffold_train, scaffold_test = scaffold_split(smiles)

    for name, tr, te in [
        ("random_split", random_train, random_test),
        ("scaffold_split", scaffold_train, scaffold_test),
    ]:
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        rf, gb = fit_regressors(Xtr, y[tr])
        p_rf = rf.predict(Xte)
        p_gb = gb.predict(Xte)
        p = (p_rf + p_gb) / 2
        results[name] = {
            **regression_metrics(y[te], p_rf, p_gb, y[tr]),
            **split_report(smiles, tr, te),
        }
        save_regression_plots(y[te], p, y[te] - p, f"solubility_{name}")

    scaler = StandardScaler().fit(X[scaffold_train])
    rf, gb = fit_regressors(scaler.transform(X[scaffold_train]), y[scaffold_train])
    joblib.dump(rf, MODELS / "solubility_rf.joblib")
    joblib.dump(gb, MODELS / "solubility_gb.joblib")
    joblib.dump(scaler, MODELS / "solubility_scaler.joblib")
    return results


def evaluate_classification_splits(smiles, X, y):
    results = {}
    random_train, random_test = train_test_split(
        np.arange(len(smiles)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaffold_train, scaffold_test = scaffold_split(smiles)
    if len(np.unique(y[scaffold_test])) < 2 or len(np.unique(y[scaffold_train])) < 2:
        raise ValueError("Scaffold split has only one class in train/test; use a different seed or larger dataset.")

    for name, tr, te in [
        ("random_split", random_train, random_test),
        ("scaffold_split", scaffold_train, scaffold_test),
    ]:
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        rf, gb = fit_classifiers(Xtr, y[tr])
        p_rf = rf.predict_proba(Xte)[:, 1]
        p_gb = gb.predict_proba(Xte)[:, 1]
        p = (p_rf + p_gb) / 2
        results[name] = {
            **classification_metrics(y[te], p_rf, p_gb, y[tr]),
            **split_report(smiles, tr, te),
            "positive_rate_test": round(float(y[te].mean()), 4),
        }
        save_classification_plots(y[te], p, f"toxicity_{name}")

    scaler = StandardScaler().fit(X[scaffold_train])
    rf, gb = fit_classifiers(scaler.transform(X[scaffold_train]), y[scaffold_train])
    joblib.dump(rf, MODELS / "toxicity_rf.joblib")
    joblib.dump(gb, MODELS / "toxicity_gb.joblib")
    joblib.dump(scaler, MODELS / "toxicity_scaler.joblib")
    return results


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_solubility():
    print("=== Solubility: random vs scaffold evaluation ===")
    data_path = BASE / "data" / "delaney.csv"
    df = pd.read_csv(data_path)
    smiles_all = df["smiles"].tolist()
    y_all = df["measured log solubility in mols per litre"].astype(float).values
    X, keep = build_feature_matrix(smiles_all)
    smiles = [smiles_all[i] for i in keep]
    y = y_all[keep]
    results = evaluate_regression_splits(smiles, X, y)
    return {
        "task": "regression",
        "dataset": "ESOL (Delaney)",
        "dataset_sha256": file_sha256(data_path),
        "n_molecules": int(len(smiles)),
        "features": DESCRIPTOR_NAMES,
        "deployment_split": "scaffold_split",
        "evaluation": results,
    }


def train_toxicity():
    print("=== Tox21 NR-AR: random vs scaffold evaluation ===")
    data_path = BASE / "data" / "tox21.csv"
    df = pd.read_csv(data_path)[["smiles", "NR-AR"]].dropna()
    smiles_all = df["smiles"].tolist()
    y_all = df["NR-AR"].astype(int).values
    X, keep = build_feature_matrix(smiles_all)
    smiles = [smiles_all[i] for i in keep]
    y = y_all[keep]
    results = evaluate_classification_splits(smiles, X, y)
    return {
        "task": "classification",
        "dataset": "Tox21 NR-AR",
        "dataset_sha256": file_sha256(data_path),
        "n_molecules": int(len(smiles)),
        "assay": "NR-AR",
        "features": DESCRIPTOR_NAMES,
        "deployment_split": "scaffold_split",
        "evaluation": results,
    }


if __name__ == "__main__":
    run_timestamp = datetime.now(timezone.utc).isoformat()
    sol = train_solubility()
    tox = train_toxicity()
    metrics = {
        "version": "3.1.0",
        "generated_at_utc": run_timestamp,
        "random_state": RANDOM_STATE,
        "requested_test_fraction": TEST_SIZE,
        "solubility": sol,
        "toxicity": tox,
        "evaluation_artifacts": "models/evaluation/",
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODELS / "model_version.json").write_text(json.dumps({
        "version": "3.1.0",
        "generated_at_utc": run_timestamp,
        "training_script": "train_models.py",
        "random_state": RANDOM_STATE,
        "requested_test_fraction": TEST_SIZE,
        "deployment_split": "Bemis-Murcko scaffold",
        "artifacts": [
            "solubility_rf.joblib", "solubility_gb.joblib", "solubility_scaler.joblib",
            "toxicity_rf.joblib", "toxicity_gb.joblib", "toxicity_scaler.joblib",
        ],
    }, indent=2))
    print(json.dumps(metrics, indent=2))
