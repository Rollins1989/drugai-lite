# DrugAI Lite — Explainable Computational Drug-Discovery Workspace

DrugAI Lite is a working, deployable computational drug-discovery prototype built around **RDKit + scikit-learn + FastAPI**.

It is deliberately scoped as an honest research/portfolio system: the ML predictions come from trained public-data models, while chemistry rules and similarity analysis are deterministic RDKit computations.

## What the upgraded v3 does

### Single-molecule analysis
- Canonical SMILES, molecular formula and Murcko scaffold
- RDKit physicochemical descriptors: MolWt, LogP, TPSA, HBD/HBA, rotatable bonds, aromatic rings, QED, etc.
- Solubility prediction from a Random Forest + Gradient Boosting ensemble
- Tox21 NR-AR toxicity probability from a Random Forest + Gradient Boosting ensemble
- Random-split vs Bemis–Murcko scaffold-split evaluation
- ROC-AUC, PR-AUC, precision, recall, F1, RMSE, MAE and R² reporting
- Deployment models trained on the scaffold split to avoid scaffold leakage
- Target-specific IC50/pIC50 prediction pipeline using ChEMBL
- Model-disagreement confidence signals
- Lipinski Rule-of-Five checks
- Veber oral-bioavailability heuristics (rotatable bonds / TPSA)
- PAINS structural-alert screening using RDKit FilterCatalog
- Morgan fingerprint nearest-neighbour search against the bundled ESOL/Tox21 reference chemistry
- A simple applicability-domain signal based on nearest Tanimoto similarity
- Descriptor-importance explanations for both trained models
- Composite screening score combining toxicity, solubility, QED, chemical-space proximity, structural alerts and rule checks

### Batch virtual screening
Upload `.smi`, `.txt`, or `.csv` files (CSV requires a `smiles` column).

The funnel performs:

`validity → deduplication → Lipinski → PAINS flagging → ML scoring → chemical-space signal → ranking`

The API returns the top 20 candidates with structures, model outputs and screening metadata.

## What is real vs. roadmap

| Component | Status |
|---|---|
| RDKit descriptors / structures | Real |
| Solubility ensemble | Real, trained on ESOL/Delaney |
| Toxicity ensemble | Real, trained on Tox21 NR-AR |
| Model disagreement | Real RF vs GB spread |
| Lipinski / Veber | Real deterministic rules |
| PAINS alerts | Real RDKit FilterCatalog |
| Morgan/Tanimoto analog search | Real against bundled reference datasets |
| Applicability-domain signal | Real nearest-neighbour similarity heuristic |
| Batch virtual screening | Real |
| Protein-ligand docking | Roadmap |
| Target-specific IC50/pIC50 baseline | Real |
| Knowledge graph / literature RAG | Roadmap |
| Molecular generation | Roadmap |
| Production auth / queues / multi-tenancy | Roadmap |

## Target-specific activity

The first target-specific baseline is **EGFR (CHEMBL203)** using ChEMBL IC50 measurements. The training script downloads human EGFR IC50 records, keeps exact measurements, aggregates repeated measurements per canonical SMILES, converts ChEMBL pChEMBL values to a single activity label, and trains Morgan-fingerprint Random Forest + Gradient Boosting regressors using a Bemis–Murcko scaffold split.

ChEMBL is a curated public bioactivity resource containing compound-target activity measurements.

Run:

```powershell
cd backend
python train_target_activity.py --target CHEMBL203 --name egfr --max-records 5000
```

This creates:

```text
backend/data/targets/egfr_ic50.csv
backend/models/targets/egfr/
    rf.joblib
    gb.joblib
    scaler.joblib
    metrics.json
```

The API then exposes:

- `GET /api/activity-targets` — installed target models and evaluation metrics
- `POST /api/predict-activity` — target-specific pIC50 prediction

Example:

```json
POST /api/predict-activity
{"smiles":"COc1ccc2nc(NC3CCN(CC3)C)nc2c1", "target":"egfr"}
```

The model is a **research prediction**, not an experimental binding measurement.

## Models

### Solubility
ESOL/Delaney dataset, using the bundled `data/delaney.csv`. Features are the 12 descriptors used by the training script. Two models are trained:
- Random Forest Regressor
- Gradient Boosting Regressor

The ensemble is the arithmetic mean of both predictions.

### Toxicity
Tox21 `NR-AR` assay, using the bundled `data/tox21.csv`.
- Random Forest Classifier
- Gradient Boosting Classifier

The ensemble is the arithmetic mean of both positive-class probabilities.

Run `python train_models.py` to reproduce the model artifacts and metrics. The final deployed solubility/toxicity models are trained on the scaffold split, while `models/metrics.json` retains both random-split and scaffold-split results so the generalization gap is visible.

## API

- `GET /api/health` — service and model metrics
- `GET /api/model-cards` — model transparency metadata
- `POST /api/analyze` — single SMILES analysis
- `POST /api/screen` — batch screening upload

Example:

```json
POST /api/analyze
{"smiles":"CC(=O)Oc1ccccc1C(=O)O"}
```

## Run locally

```powershell
cd backend
python -m pip install -r requirements.txt
python train_models.py
python -m uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`.

## Docker

```powershell
cd backend
docker build -t drugai-lite .
docker run --rm -p 8000:8000 drugai-lite
```

## Project structure

```text
DrugAI-Lite/
├── README.md
└── backend/
    ├── main.py
    ├── train_models.py
    ├── requirements.txt
    ├── Dockerfile
    ├── data/
    │   ├── delaney.csv
    │   └── tox21.csv
    ├── models/
    │   ├── *.joblib
    │   └── metrics.json
    └── static/
        ├── index.html
        ├── app.js
        └── style.css
```

## Important limitation

DrugAI Lite is a computational research aid. A predicted toxicity probability is **not** a clinical toxicity assessment, and the composite score is a prioritization heuristic rather than a drug-development decision.
