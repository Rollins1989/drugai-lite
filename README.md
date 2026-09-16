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
| Testing | pytest |
| Packaging/deployment | Docker |
| CI | GitHub Actions |

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

Reported metrics include:

- ROC-AUC
- PR-AUC
- Accuracy
- Precision
- Recall
- F1

ROC-AUC and PR-AUC are emphasized because class imbalance makes accuracy alone insufficient.

### 4. Target-specific activity prediction

DrugAI Lite includes a target-specific activity workflow for:

**EGFR — CHEMBL203**

The training pipeline:

```text
ChEMBL activity records
        ↓
IC50 filtering
        ↓
Human-target filtering
        ↓
Molecule-level aggregation
        ↓
Morgan fingerprints
        ↓
Bemis-Murcko scaffold split
        ↓
Random Forest + Gradient Boosting
        ↓
pIC50 prediction
        ↓
IC50 in nM
```

The target model reports:

- Number of molecules
- Train/test counts
- Scaffold counts
- Scaffold overlap
- R²
- RMSE
- MAE

### 5. Virtual screening

Upload either:

- `.csv` with a `smiles` column
- `.txt` containing one SMILES per line
- `.smi` containing one SMILES per line

The screening funnel:

```text
Input library
   ↓
SMILES validation
   ↓
Canonicalization / deduplication
   ↓
Lipinski filtering
   ↓
PAINS flagging
   ↓
Solubility + toxicity prediction
   ↓
Chemical-space proximity
   ↓
Heuristic screening score
   ↓
Ranked candidate list
```

The API limits a batch request to **5,000 molecules**.

### 6. Chemical-space and structural analysis

The application also provides:

- Morgan fingerprint similarity
- Tanimoto nearest-neighbour search
- Applicability-domain signal based on nearest-neighbour similarity
- Bemis-Murcko scaffold identification
- PAINS structural-alert screening
- Lipinski rule checks
- Veber rule checks

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

These metrics describe the bundled evaluation artifacts documented with the project. They should not be interpreted as evidence of clinical or experimental efficacy.

---

## Why scaffold splitting?

Random molecular splits can place closely related structures in both training and test sets. This can make generalization look stronger than it is for genuinely novel chemical scaffolds.

DrugAI Lite therefore also uses **Bemis-Murcko scaffold splitting** and reports scaffold overlap as a leakage diagnostic.

The deployed solubility and toxicity model artifacts are trained using the scaffold split, while both random and scaffold evaluations are retained for comparison.

---

## Example prediction

Example EGFR input:

```text
COC1=C(OCCCN2CCCCC2)C=CC(=C1)NC3=NC=CC(=C3)C#N
```

Example output:

```text
Target: EGFR (CHEMBL203)
Predicted pIC50: 5.668
Predicted IC50: 2148.69 nM
Random Forest: 5.519
Gradient Boosting: 5.817
```

The application also exposes model disagreement as a heuristic uncertainty signal. It is **not calibrated predictive confidence**.

---

## API

Once the server is running, FastAPI provides interactive documentation at:

```text
http://127.0.0.1:8000/docs
```

### Health check

```bash
curl http://127.0.0.1:8000/api/health
```

### Analyze a molecule

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"smiles":"CC(=O)Oc1ccccc1C(=O)O"}'
```

### List target models

```bash
curl http://127.0.0.1:8000/api/activity-targets
```

### Predict target activity

```bash
curl -X POST http://127.0.0.1:8000/api/predict-activity \
  -H "Content-Type: application/json" \
  -d '{"smiles":"CCO","target":"egfr"}'
```

### Batch screening

```bash
curl -X POST http://127.0.0.1:8000/api/screen \
  -F "file=@molecules.csv"
```

### Model cards

```bash
curl http://127.0.0.1:8000/api/model-cards
```

---

## Web interface

The repository includes a browser interface served by FastAPI with four primary views:

- **Analyze** — single-molecule chemistry and ML analysis
- **Screen** — batch virtual screening
- **Target Activity** — target-specific prediction
- **Model Cards** — model transparency and evaluation information

The frontend source lives in [`backend/static/`](backend/static/).

For local use, start the API and open:

```text
http://127.0.0.1:8000/
```

---

## Project structure

```text
drugai-lite/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── backend/
│   ├── data/
│   │   ├── delaney.csv
│   │   ├── tox21.csv
│   │   └── targets/
│   │
│   ├── models/
│   │   ├── solubility_*.joblib
│   │   ├── toxicity_*.joblib
│   │   ├── metrics.json
│   │   └── targets/
│   │
│   ├── static/
│   │   ├── index.html
│   │   ├── app.js
│   │   └── style.css
│   │
│   ├── tests/
│   │   └── test_api.py
│   │
│   ├── main.py
│   ├── train_models.py
│   ├── train_target_activity.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── docs/
│   └── architecture.svg
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

### 1. Clone

```bash
git clone https://github.com/Rollins1989/drugai-lite.git
cd drugai-lite
```

### 2. Create a virtual environment

```bash
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

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Start the application

```bash
cd backend
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Training

### Property and toxicity models

From `backend/`:

```bash
python train_models.py
```

This evaluates random and scaffold splits and saves the scaffold-trained deployment models plus `models/metrics.json`.

### EGFR target model

```bash
python train_target_activity.py \
  --target CHEMBL203 \
  --name egfr \
  --max-records 5000
```

The target pipeline retrieves eligible IC50 records from ChEMBL, aggregates repeated measurements at molecule level, calculates Morgan fingerprints, performs a scaffold split, trains two regressors, and saves the target-specific model bundle.

> Because ChEMBL is an external and evolving database, a future training run may produce different data and metrics. Record the retrieval date and source release when producing new model artifacts.

---

## Testing

Run the current API test suite from `backend/`:

```bash
python -m pytest tests
```

GitHub Actions runs the same test suite automatically for pushes and pull requests targeting `main`.

---

## Docker

Build the image:

```bash
docker build -t drugai-lite backend
```

Run it:

```bash
docker run --rm -p 8000:8000 drugai-lite
```

Then open:

```text
http://127.0.0.1:8000/
```

---

## Data sources & attribution

DrugAI Lite uses public scientific datasets and external scientific databases for model development.

### ESOL / Delaney

Used for aqueous solubility regression.

- Dataset: Delaney ESOL
- Target: measured log solubility in mol/L
- Local copy: `backend/data/delaney.csv`

### Tox21

Used for binary toxicity classification.

- Dataset: Tox21
- Assay used: `NR-AR`
- Local copy: `backend/data/tox21.csv`
- Interpretation: androgen receptor nuclear-signalling assay endpoint

### ChEMBL

Used for target-specific activity modeling.

- Target: EGFR / `CHEMBL203`
- Endpoint: IC50
- Model target representation: pIC50
- Source: ChEMBL activity API
- Training script: `backend/train_target_activity.py`

When redistributing or extending the datasets, check the respective source's current terms, attribution requirements, and release/version information.

---

## Scientific limitations

DrugAI Lite is intended for learning, experimentation, and portfolio demonstration.

Important limitations include:

- Predictions are computational estimates, not experimental measurements.
- Model performance depends on the training data and molecular representation.
- EGFR is a target-specific model and is not a universal target-activity predictor.
- Scaffold evaluation improves the assessment of chemical generalization but does not eliminate all sources of dataset bias.
- Model disagreement is used as a heuristic uncertainty signal and is not calibrated probability.
- Applicability-domain status is based on nearest-neighbour similarity and should not be treated as a formal guarantee of prediction reliability.
- Lipinski, Veber, and PAINS checks are rule-based screens, not clinical safety determinations.
- Dataset bias, assay variability, measurement noise, and chemical-space limitations can affect generalization.
- Experimental validation remains necessary for any scientific or drug-development decision.

---

## Current version

**v3**

Current focus:

- Scaffold-aware evaluation
- Target-specific activity prediction
- EGFR activity modeling
- FastAPI prediction endpoints
- Interactive molecular analysis
- Batch virtual screening
- Model transparency

---

## Future development

Potential next steps include:

- Calibrated uncertainty estimation
- Cross-validation and repeated scaffold splits
- Residual and calibration analysis
- Additional target models
- Stronger chemical ML baselines
- SHAP or related explanation methods
- Model/artifact versioning
- Expanded API test coverage
- Production-oriented service modularization

---

## Author

**Kuldeep**

GitHub: [Rollins1989](https://github.com/Rollins1989)

Repository: [DrugAI Lite](https://github.com/Rollins1989/drugai-lite)

---

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE).
