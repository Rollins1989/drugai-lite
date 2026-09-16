# DrugAI Lite

### Explainable ML Platform for Computational Drug Discovery

DrugAI Lite is a research and portfolio prototype that combines **cheminformatics, machine learning, and scientific software engineering** into an interactive molecular-analysis and virtual-screening workspace.

Given a molecule as a SMILES string, the platform can calculate molecular properties, run ensemble ML predictions, identify structural alerts, search for similar reference molecules, inspect the molecule's scaffold, and expose the results through a **FastAPI backend and browser interface**.

> **Research prototype:** DrugAI Lite is not a clinical, diagnostic, safety, or regulatory system. All ML outputs are computational estimates and require experimental validation.

[![CI](https://github.com/Rollins1989/drugai-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/Rollins1989/drugai-lite/actions/workflows/ci.yml)

---

## What the project demonstrates

| Area | Implementation |
|---|---|
| Cheminformatics | RDKit, molecular descriptors, Morgan fingerprints, Murcko scaffolds |
| Molecular property prediction | ESOL/Delaney solubility regression |
| Toxicology ML | Tox21 NR-AR classification |
| Target activity | EGFR / CHEMBL203 IC50 → pIC50 prediction |
| Model evaluation | Random split + Bemis-Murcko scaffold split |
| Explainability | Descriptor feature importance + model disagreement |
| Chemical-space analysis | Morgan/Tanimoto nearest-neighbour search |
| Structural filters | Lipinski, Veber, PAINS |
| ML serving | FastAPI + Pydantic |
| Frontend | HTML, CSS, JavaScript |
| Testing | pytest + service/API integration tests |
| Packaging/deployment | Docker + runtime healthcheck |
| CI | GitHub Actions with Docker smoke test |
| Architecture | Layered API, chemistry, model, service, and schema modules |

---

## Core workflow

```text
                         ┌──────────────────────┐
                         │   SMILES molecule    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      RDKit parse     │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
           Descriptors         Fingerprints       Structure
                  │                 │                 │
          ┌───────┴───────┐         │          ┌──────┴──────┐
          ▼               ▼         ▼          ▼             ▼
     Solubility       Toxicity   Analog      Scaffold    Alerts/rules
       ensemble        ensemble   search     analysis    Lipinski/Veber/PAINS
          │               │         │            │             │
          └───────────────┴─────────┴────────────┴─────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Screening results  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     FastAPI API       │
                         │   + browser frontend  │
                         └──────────────────────┘
```

A rendered architecture diagram is also available at [`docs/architecture.svg`](docs/architecture.svg).

---

## Features

### 1. Molecular analysis

Enter a SMILES string and obtain:

- Molecular formula and canonical SMILES
- Molecular weight and exact molecular weight
- LogP and TPSA
- Hydrogen-bond donors/acceptors
- Rotatable bonds and aromatic rings
- Fraction CSP3 and heteroatom count
- QED and other molecular descriptors
- Murcko scaffold
- 2D molecular structure rendering

### 2. Solubility prediction

The ESOL/Delaney dataset is used to train regression models from RDKit molecular descriptors.

Models:

- Random Forest Regressor
- Gradient Boosting Regressor
- Ensemble mean prediction

Metrics:

- R²
- RMSE
- MAE

Both conventional random splitting and scaffold-based evaluation are reported.

### 3. Toxicity prediction

The project uses the **Tox21 NR-AR assay** as a binary classification task.

Models:

- Random Forest Classifier
- Gradient Boosting Classifier
- Ensemble probability

Reported metrics include ROC-AUC, PR-AUC, Accuracy, Precision, Recall, and F1. ROC-AUC and PR-AUC are emphasized because class imbalance makes accuracy alone insufficient.

### 4. Target-specific activity prediction

DrugAI Lite includes a target-specific activity workflow for **EGFR — CHEMBL203**.

The training pipeline is:

```text
ChEMBL activity records → IC50 filtering → human-target filtering
→ molecule-level aggregation → Morgan fingerprints
→ Bemis-Murcko scaffold split → RF + GB
→ pIC50 prediction → IC50 in nM
```

### 5. Virtual screening

Upload `.csv` with a `smiles` column, `.txt`, or `.smi` containing one SMILES per line. The screening API validates, canonicalizes, deduplicates, filters, predicts, compares chemical space, and returns a ranked top-20 candidate list.

Batch limit: **5,000 molecules**. HTTP upload limit: **10 MB**.

### 6. Engineering architecture

Phase 4 separates the application into focused layers rather than keeping all behavior in one FastAPI module:

```text
backend/
├── api/routes.py       # HTTP endpoints
├── chemistry.py        # RDKit operations
├── config.py           # paths and runtime configuration
├── model_service.py    # model loading and inference
├── services.py         # workflow orchestration
├── schemas.py          # Pydantic request models
└── main.py             # application assembly
```

See [`docs/phase-4-engineering.md`](docs/phase-4-engineering.md) for the architecture and testing rationale.

---

## Model evaluation

### Solubility

| Split | Ensemble R² | RMSE | MAE |
|---|---:|---:|---:|
| Random | 0.8781 | 0.7589 | 0.5193 |
| Scaffold | 0.8667 | 0.8360 | 0.6053 |

### Tox21 NR-AR

| Split | Ensemble ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| Random | 0.7692 | 0.4544 | 0.5106 |
| Scaffold | 0.7758 | 0.4626 | 0.5135 |

### EGFR / CHEMBL203

| Metric | Result |
|---|---:|
| Molecules | 3,970 |
| Test molecules | 799 |
| Scaffold overlap | 0 |
| Ensemble R² | 0.6733 |
| Ensemble RMSE | 0.7535 |
| Ensemble MAE | 0.5884 |

These are bundled evaluation results; they are not evidence of clinical or prospective experimental efficacy. Phase 3 training also generates baseline and diagnostic artifacts when the training script is run.

---

## API

Run locally and open FastAPI's interactive documentation at `http://127.0.0.1:8000/docs`.

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/version
curl http://127.0.0.1:8000/api/model-cards
curl http://127.0.0.1:8000/api/activity-targets
```

Analyze a molecule:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"smiles":"CC(=O)Oc1ccccc1C(=O)O"}'
```

Batch screen:

```bash
curl -X POST http://127.0.0.1:8000/api/screen -F "file=@molecules.csv"
```

---

## Web interface

The browser interface is served by FastAPI and includes:

- **Analyze** — single-molecule chemistry and ML analysis
- **Screen** — batch virtual screening
- **Target Activity** — target-specific prediction
- **Model Cards** — model transparency and evaluation information

Frontend source: [`backend/static/`](backend/static/).

---

## Project structure

```text
drugai-lite/
├── .github/workflows/ci.yml
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── data/
│   │   ├── delaney.csv
│   │   ├── tox21.csv
│   │   └── targets/
│   ├── models/
│   │   ├── solubility_*.joblib
│   │   ├── toxicity_*.joblib
│   │   ├── metrics.json
│   │   └── targets/
│   ├── static/
│   ├── tests/
│   │   ├── test_api.py
│   │   └── test_services.py
│   ├── chemistry.py
│   ├── config.py
│   ├── model_service.py
│   ├── schemas.py
│   ├── services.py
│   ├── main.py
│   ├── train_models.py
│   ├── train_target_activity.py
│   ├── requirements.txt
│   └── Dockerfile
├── docs/
│   ├── architecture.svg
│   ├── model-evaluation.md
│   └── phase-4-engineering.md
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Rollins1989/drugai-lite.git
cd drugai-lite
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install backend dependencies:

```bash
cd backend
python -m pip install -r requirements.txt
```

Start the application:

```bash
uvicorn main:app --reload
```

### Docker

```bash
cd backend
docker build -t drugai-lite .
docker run --rm -p 8000:8000 drugai-lite
```

The container exposes a healthcheck at `/api/health`.

---

## Training and evaluation

Solubility/toxicity training and diagnostics:

```bash
cd backend
python train_models.py
```

Target-specific EGFR training:

```bash
python train_target_activity.py --target CHEMBL203 --name egfr
```

Phase 3 evaluation details are documented in [`docs/phase-3-evaluation.md`](docs/phase-3-evaluation.md).

---

## Scientific limitations

- No external validation or prospective experimental validation is claimed.
- Ensemble disagreement is a model-agreement signal, not calibrated predictive confidence.
- The screening score is a hand-weighted heuristic prioritization score, not a validated efficacy/safety/developability endpoint.
- Applicability-domain output is a nearest-neighbour chemical-space proximity signal, not a formal statistical applicability-domain guarantee.
- Tree feature importance is global model-level importance, not causal per-molecule attribution.
- Tox21 NR-AR represents one assay endpoint and should not be generalized to overall human toxicity.
- ChEMBL target activity measurements can contain assay and experimental heterogeneity.
- All predictions are intended for research/hypothesis generation and require experimental validation.

---

## License

MIT License. See [`LICENSE`](LICENSE).

## Version

Application architecture: **3.0.0 (Phase 4)**.

Model artifacts are versioned separately through `backend/models/model_version.json` when generated by the training pipeline.
