# DrugAI Lite

<p align="center">
  <strong>Computational Drug Discovery & Molecular Screening Platform</strong><br>
  <sub>RDKit • Machine Learning • Chemical Space • Virtual Screening • FastAPI • Docker</sub>
</p>

<p align="center">
  <a href="https://github.com/Rollins1989/drugai-lite/actions/workflows/ci.yml"><img src="https://github.com/Rollins1989/drugai-lite/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/Rollins1989/drugai-lite/actions/workflows/security.yml"><img src="https://github.com/Rollins1989/drugai-lite/actions/workflows/security.yml/badge.svg" alt="Security"></a>
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/RDKit-2026.03.6-blueviolet" alt="RDKit">
  <img src="https://img.shields.io/badge/FastAPI-0.141.1-009688" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-supported-2496ED" alt="Docker">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

DrugAI Lite is an open-source computational drug-discovery workspace for **molecular analysis, property prediction, target-activity modeling, chemical-space exploration, virtual screening, and scientific reporting**.

It combines **RDKit**, classical machine learning, a versioned FastAPI service, interactive browser tooling, reproducible evaluation workflows, Docker, and operational telemetry in one research-oriented platform.

> **Scientific boundary**
>
> DrugAI Lite is a computational research and prioritization tool. Its predictions, scores, applicability-domain signals, and uncertainty diagnostics are **not** clinical, regulatory, safety, efficacy, binding-affinity, or experimental evidence.

---

## Why this project?

Drug discovery models are more useful when the workflow around them is transparent, reproducible, and testable.

DrugAI Lite is built around:

**molecule → descriptors → prediction → diagnostics → chemical context → prioritization → screening → report**

The platform combines cheminformatics, supervised ML, chemical-space analysis, virtual screening, evaluation, API engineering, containerization, testing, and security automation.

---

## Core capabilities

### Molecular analysis

- Canonical SMILES, molecular formula, exact mass, and 2D structure
- Molecular weight, LogP, TPSA, HBD/HBA, rotatable bonds, rings, CSP3, QED
- Bemis–Murcko scaffold
- Lipinski and Veber checks
- PAINS structural alerts

### Machine learning

- ESOL / Delaney solubility regression
- Tox21 NR-AR classification
- Optional target-specific pIC50 modeling pipeline
- Random Forest + Gradient Boosting ensembles
- RF/GB agreement diagnostics
- Empirical RF-tree spread diagnostics
- Global feature importance
- Local ±1 training-standard-deviation sensitivity diagnostics
- Model and application provenance

### Chemical-space analysis

- Morgan fingerprints
- Tanimoto similarity
- Nearest-neighbour search
- PCA projection
- Agglomerative clustering
- Interactive chemical-space visualization
- Custom reference-library upload
- Dataset quality reporting
- Pareto multi-objective ranking

### Virtual screening

- CSV / TXT / SMI ingestion
- SMILES validation and canonicalization
- Duplicate detection
- Metadata preservation
- Lipinski / Veber / PAINS filtering
- ML predictions
- Optional target-aware pIC50 prediction
- Candidate prioritization
- Asynchronous screening jobs
- Ranked CSV export

### Scientific reporting

- Molecular PDF reports
- Model cards
- Evaluation metadata
- Runtime/model provenance
- Request IDs and processing latency
- Prometheus-compatible operational metrics
- Docker deployment
- GitHub Actions, CodeQL, and Dependabot

---

## Model performance

The bundled deployment models include a **legacy evaluation snapshot**.

| Model | Metric | Benchmark |
|---|---:|---:|
| ESOL ensemble | R² | ~0.878 |
| ESOL ensemble | RMSE | ~0.759 |
| Tox21 NR-AR ensemble | ROC-AUC | ~0.769 |

These numbers describe the bundled deployment benchmark snapshot. They are **not** prospective performance claims and are not evidence of biological or clinical validity.

The reproducible training pipeline can regenerate evaluation artifacts using random and Bemis–Murcko scaffold-disjoint evaluation, with regression/classification metrics and diagnostic plots.

See [docs/evaluation.md](docs/evaluation.md).

---

## Architecture

```text
                           DrugAI Lite
                                │
                ┌───────────────┴───────────────┐
                │                               │
          Browser Workspace                 REST API
                │                               │
                └───────────────┬───────────────┘
                                │
                            FastAPI
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
     RDKit /                 ML Models              Analytics
   Cheminformatics          RF + GB              PCA / Clustering
        │                  ESOL / Tox21             Similarity
   Descriptors             Target pipeline            Pareto
   Lipinski
   PAINS
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                     Evaluation + Diagnostics
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
             Docker / CI                  Prometheus
          CodeQL / Dependabot              Metrics
```

---

## API

Interactive OpenAPI documentation is available at:

`http://127.0.0.1:8000/docs`

New integrations should use **`/api/v1`**. The original **`/api`** namespace remains available for backward compatibility.

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/analyze` | Single-molecule analysis |
| `POST /api/v1/compare` | Compare two molecules |
| `POST /api/v1/predict-activity` | Target-specific activity |
| `POST /api/v1/screen` | Synchronous screening |
| `POST /api/v1/screen/jobs` | Asynchronous screening |
| `GET /api/v1/screen/jobs/{job_id}` | Screening job status |
| `POST /api/v1/chemical-space` | PCA + clustering |
| `POST /api/v1/pareto` | Pareto ranking |
| `POST /api/v1/dataset/quality` | Dataset QC |
| `POST /api/v1/references/upload` | Reference-library upload |
| `POST /api/v1/screen/export.csv` | Screening CSV export |
| `POST /api/v1/analyze/report.pdf` | Molecular PDF report |
| `GET /api/v1/model-cards` | Model metadata |
| `GET /api/v1/evaluation` | Evaluation metadata |
| `GET /api/v1/metrics` | Operational metrics |
| `GET /api/v1/version` | Runtime/model provenance |
| `GET /api/v1/health` | Health information |

### Example

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"smiles":"CC(=O)Oc1ccccc1C(=O)O"}'
```

---

## Screening philosophy

The prioritization number is deliberately labeled a **heuristic candidate-prioritization score**.

It combines computational signals such as modeled toxicity, solubility, QED, reference-space proximity, PAINS status, and drug-likeness rules.

It is **not** a validated efficacy score, clinical risk score, binding-affinity score, developability guarantee, or replacement for experimental testing.

For target-enabled screening, the Pareto layer can consider activity, toxicity, solubility, and QED simultaneously rather than hiding every objective behind one number.

---

## Uncertainty and applicability domain

DrugAI Lite separates diagnostic signals from calibrated uncertainty.

| Signal | Interpretation | Not equivalent to |
|---|---|---|
| RF/GB agreement | Model-family agreement | Probability of correctness |
| RF-tree spread | Empirical ensemble spread | Confidence interval |
| Tanimoto proximity | Similarity to reference space | Formal applicability-domain guarantee |
| Feature importance | Model explanation | Causal attribution |
| Local sensitivity | Local model-response diagnostic | Experimental sensitivity |

---

## Validation hierarchy

The project follows a layered validation philosophy:

1. Random benchmark split
2. Bemis–Murcko scaffold-disjoint split
3. Temporal split where timestamps are available
4. Independent external dataset
5. Prospective experimental validation

The current automated workflow supports the first two.

The project does **not** claim prospective experimental validation.

---

## Reproducible training

### Solubility and toxicity

```bash
cd backend
python train_models.py
```

The pipeline records dataset hashes and generates evaluation/model artifacts.

### Optional target activity

```bash
cd backend
python train_target_activity.py --target CHEMBL203 --name egfr
```

The target pipeline can retrieve current ChEMBL data, train a target-specific model, and record retrieval/query metadata and dataset hashes.

**Important:** the presence of the training pipeline does not mean an EGFR model is bundled.

Targets are exposed only when trained artifacts exist under:

```text
backend/models/targets/<name>/
```

---

## Scientific limitations

DrugAI Lite is intentionally conservative about what its models can establish.

- Tox21 NR-AR is one assay endpoint and must not be generalized to overall human toxicity.
- pIC50 values are computational estimates, not experimental measurements.
- RF/GB agreement and RF-tree spread are diagnostic signals, not calibrated uncertainty.
- Nearest-neighbour similarity is not a formal statistical applicability-domain guarantee.
- Feature importance and local sensitivity are explanatory diagnostics, not causal evidence.
- The candidate-prioritization score is hand-weighted and unvalidated.
- ChEMBL measurements can contain assay and experimental heterogeneity.
- Models can fail outside their training distribution.
- Computational prioritization does not establish biological efficacy, safety, pharmacokinetics, or clinical utility.
- Experimental validation remains necessary for biological conclusions.

---

## Installation

### Requirements

- Python 3.12+
- Git
- Docker (optional)

### Local setup

```bash
git clone https://github.com/Rollins1989/drugai-lite.git
cd drugai-lite
python -m venv .venv
```

#### Windows

```powershell
.venv\Scripts\activate
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

Install and run:

```bash
cd backend
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

## Docker

```bash
cd backend
docker build -t drugai-lite .
docker run --rm -p 8000:8000 drugai-lite
```

---

## Development

Install development dependencies:

```bash
cd backend
python -m pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest -q
```

Run Ruff:

```bash
ruff check .
```

Compile-check:

```bash
python -m compileall .
```

CI validates dependency consistency, linting, compilation, tests/coverage, and Docker build/smoke behaviour.

Security automation includes CodeQL and Dependabot.

---

## Production engineering

### API versioning

- `/api/v1` is the supported versioned API.
- `/api` is retained as a compatibility surface.
- OpenAPI operation IDs are unique across both registrations.

### Observability

`GET /api/v1/metrics` exposes Prometheus-compatible request and latency metrics.

Metrics use normalized route paths rather than raw job IDs to avoid unnecessary label cardinality.

### Async screening

The screening service includes:

- batch-size limits
- bounded active jobs
- TTL cleanup
- input-format validation
- explicit job lifecycle/status

Job state and custom reference libraries are currently **process-local**. Multi-worker or multi-instance deployment should replace these with shared persistent infrastructure and add authentication/rate limiting.

### Container security

Docker runs the application as a non-root user and includes a healthcheck.

---

## Repository structure

```text
drugai-lite/
├── .github/workflows/
│   ├── ci.yml
│   ├── security.yml
│   └── release.yml
├── backend/
│   ├── api/routes.py
│   ├── models/
│   ├── tests/
│   ├── analytics.py
│   ├── chemistry.py
│   ├── config.py
│   ├── main.py
│   ├── model_service.py
│   ├── schemas.py
│   ├── services.py
│   ├── train_models.py
│   ├── train_target_activity.py
│   └── requirements*.txt
├── docs/
│   ├── evaluation.md
│   ├── scientific-methodology.md
│   └── release-process.md
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Documentation

| Document | Purpose |
|---|---|
| [Scientific methodology](docs/scientific-methodology.md) | Methods, diagnostics, chemical-space interpretation, limitations |
| [Evaluation](docs/evaluation.md) | Reproducible evaluation and validation contract |
| [Release process](docs/release-process.md) | Release and deployment guidance |
| [Changelog](CHANGELOG.md) | Version history |
| [Citation](CITATION.cff) | Software citation metadata |

---

## Versioning

**Application:** `4.0.0`  
**Bundled model bundle:** `3.1.0`

Model manifest: [`backend/models/model_version.json`](backend/models/model_version.json)

---

## Roadmap

Potential future platform work:

- persistent experiment history
- shared distributed screening jobs
- Redis/Celery or equivalent queue infrastructure
- PostgreSQL-backed state
- authentication and authorization
- rate limiting
- user-isolated reference libraries
- model registry/version management
- independent external validation datasets
- calibrated uncertainty methods
- expanded target-specific model bundles
- experiment tracking and audit trails

These are future directions, not current capabilities.

---

## Contributing

Contributions, scientific review, reproducibility improvements, bug reports, and engineering feedback are welcome.

Before opening a pull request:

1. Run the test suite.
2. Run Ruff.
3. Confirm documentation matches implementation.
4. Avoid unsupported scientific claims.
5. Keep model limitations explicit.

---

## License

Released under the **MIT License**. See [`LICENSE`](LICENSE).

## Citation

If you use DrugAI Lite in research, teaching, experimentation, or derivative work, see [`CITATION.cff`](CITATION.cff).

---

**Computational predictions are hypotheses. Experimental validation remains the final authority.**
