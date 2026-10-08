# DrugAI Lite

### Open-source computational drug-discovery workspace

DrugAI Lite combines **RDKit cheminformatics, machine learning, target-specific activity prediction, chemical-space analysis and virtual screening** behind a FastAPI API and browser workspace.

> **Scientific boundary:** DrugAI Lite is a computational research aid. Predictions, prioritization scores, applicability-domain signals and uncertainty diagnostics are not clinical, regulatory, safety, efficacy or experimental evidence.

[![CI](https://github.com/Rollins1989/drugai-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/Rollins1989/drugai-lite/actions/workflows/ci.yml)

## Feature set

### Molecular intelligence
- Canonical SMILES, molecular formula, exact mass and 2D structure
- MW, LogP, TPSA, HBD/HBA, rotatable bonds, rings, CSP3 and QED
- Murcko scaffold
- Lipinski and Veber checks
- PAINS structural alerts

### ML prediction
- ESOL/Delaney solubility regression
- Tox21 NR-AR classification
- Optional target-specific activity prediction (EGFR/CHEMBL203 training pipeline included; model bundles are not assumed to be present)
- Random Forest + Gradient Boosting ensembles
- Random and Bemis–Murcko scaffold evaluation
- RF/GB model-agreement diagnostics
- Empirical RF-tree spread diagnostics
- Global feature importance
- Local one-feature ±1 training-standard-deviation sensitivity diagnostics

### Chemical-space intelligence
- Morgan fingerprints and Tanimoto nearest-neighbour search
- PCA chemical-space projection
- Agglomerative clustering
- Custom reference-library upload
- Dataset quality reporting
- Interactive PCA chemical-space scatter visualization
- Multi-objective Pareto ranking

### Virtual screening
- CSV / TXT / SMI ingestion
- CSV metadata preservation
- Structure validation and canonicalization
- Duplicate detection
- Lipinski / Veber / PAINS filtering
- Optional target-aware pIC50 prediction
- Candidate prioritization
- Asynchronous screening jobs
- Ranked CSV export

### Scientific reporting and engineering
- Runtime/model provenance
- Request IDs and processing latency
- Model cards
- Molecular PDF reports
- FastAPI + Pydantic
- Docker + healthcheck
- pytest + coverage + Ruff
- GitHub Actions + CodeQL + Dependabot

## API

The API exposes interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.

| Endpoint | Purpose |
|---|---|
| POST /api/analyze | Single-molecule analysis |
| POST /api/compare | Compare two molecules |
| POST /api/predict-activity | Target-specific activity |
| POST /api/screen | Synchronous library screening |
| POST /api/screen/jobs | Asynchronous screening |
| GET /api/screen/jobs/{job_id} | Screening job status |
| POST /api/chemical-space | PCA + clustering |
| POST /api/pareto | Multi-objective Pareto ranking |
| POST /api/dataset/quality | Dataset QC |
| POST /api/references/upload | Custom reference library |
| POST /api/screen/export.csv | Ranked CSV export |
| POST /api/analyze/report.pdf | Molecular PDF report |
| GET /api/model-cards | Model metadata and limitations |
| GET /api/version | Runtime/model provenance |
| GET /api/health | Health information |

Example:

~~~bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"smiles":"CC(=O)Oc1ccccc1C(=O)O"}'
~~~

## Screening philosophy

The prioritization number is deliberately labeled a **heuristic candidate-prioritization score**. It combines modeled toxicity, solubility category, QED, reference-space proximity, PAINS status and drug-likeness rules.

It is **not** a validated efficacy score, clinical risk score, binding-affinity score, developability guarantee, or replacement for experimental testing.

For target-enabled screening, the Pareto layer can consider activity, toxicity, solubility and QED simultaneously rather than hiding every objective behind one number.

## Uncertainty and applicability domain

DrugAI Lite distinguishes diagnostic signals from calibrated uncertainty.

- **Model agreement:** RF vs Gradient Boosting disagreement.
- **Empirical ensemble spread:** RF-tree 10th–90th percentile spread.
- **Applicability-domain signal:** nearest-neighbour Morgan/Tanimoto proximity to the selected reference space.

None of these should be interpreted as a calibrated probability of correctness.

## Evaluation

The bundled deployment models include a legacy evaluation snapshot in `backend/models/metrics.json`. Its original split metadata was not retained, so those headline numbers are not presented as a reproducible scaffold-split result. Run `backend/train_models.py` to regenerate the full random/scaffold evaluation artifacts and a fresh model manifest. Target-specific activity is optional: a cloned repository only exposes targets for which trained model artifacts are actually present.

The project does **not** claim prospective experimental validation.

Recommended validation hierarchy:
1. Random benchmark split
2. Scaffold-disjoint split
3. Temporal split where timestamps are available
4. Independent external dataset
5. Prospective experimental validation

## Installation

~~~bash
git clone https://github.com/Rollins1989/drugai-lite.git
cd drugai-lite
python -m venv .venv
~~~

Windows:
~~~text
.venv\\Scripts\\activate
~~~

macOS/Linux:
~~~bash
source .venv/bin/activate
~~~

Install and run:
~~~bash
cd backend
python -m pip install -r requirements.txt
uvicorn main:app --reload
~~~

Open `http://127.0.0.1:8000`.

### Docker
~~~bash
cd backend
docker build -t drugai-lite .
docker run --rm -p 8000:8000 drugai-lite
~~~

## Training

Solubility and toxicity:
~~~bash
cd backend
python train_models.py
~~~

Target activity (optional; downloads current ChEMBL data and creates a target model bundle):
~~~bash
python train_target_activity.py --target CHEMBL203 --name egfr
~~~

The EGFR/CHEMBL203 training pipeline is included, but no target model is treated as bundled unless `backend/models/targets/<name>/` contains the required artifacts. The UI/API therefore report zero available targets rather than advertising an unavailable EGFR model.

The target pipeline records ChEMBL retrieval/query metadata and dataset hashes so refreshed training runs are auditable.

## Scientific methodology

See [`docs/scientific-methodology.md`](docs/scientific-methodology.md) for the model inputs, validation hierarchy, chemical-space interpretation, sensitivity methodology and scientific boundaries.

## Scientific limitations
- Tox21 NR-AR is one assay endpoint and must not be generalized to overall human toxicity.
- pIC50 predictions are computational estimates, not experimental activity measurements.
- RF/GB agreement and RF-tree spread are diagnostic signals, not calibrated confidence intervals.
- Nearest-neighbour chemical-space proximity is not a formal statistical applicability-domain guarantee.
- Global feature importance and local ±1 training-standard-deviation sensitivity are explanatory diagnostics, not causal attribution.
- The candidate-prioritization score is hand-weighted and unvalidated.
- ChEMBL measurements can contain assay and experimental heterogeneity.
- Models can fail outside their training distribution.
- Experimental validation remains necessary for biological conclusions.

## Version
**Application: 4.0.0**  
**Bundled model bundle: 3.1.0 (explicitly tracked in `backend/models/model_version.json`)**  
**Software citation: `CITATION.cff`**

## License
MIT License. See LICENSE.