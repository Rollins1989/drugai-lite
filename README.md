# DrugAI Lite — Computational Screening Workspace

A working, deployable slice of a computational drug-discovery platform:
molecular descriptor computation, trained ML models for solubility and
toxicity prediction, ensemble-based confidence, drug-likeness scoring,
and a real virtual-screening funnel over batches of molecules.

Live demo flow: type a SMILES → get RDKit descriptors, a solubility
prediction, a toxicity prediction, model confidence (from RF/GB
disagreement), Lipinski/QED drug-likeness, and a rendered 2D structure —
all computed on the fly, nothing hardcoded. Upload a file of SMILES →
get a real funnel (validity → Lipinski filter → ML scoring → ranked
top candidates).

## Why this scope, and not the full vision

The original concept behind this project (see `docs/full-vision.md` if
you keep the source doc) includes protein-ligand docking, GNN activity
models against specific targets, a biomedical knowledge graph, an
autonomous research agent, and full multi-tenant SaaS infrastructure.
Building that for real — not mocked — is a multi-engineer, multi-month
effort requiring licensed structural biology data, GPU training
infrastructure, and a production data pipeline.

This build deliberately does the smaller thing *honestly* rather than
the larger thing *superficially*: every prediction here comes from a
model that was actually trained and evaluated on public data (metrics
below), not from a lookup table or a scripted response. That's the
version that survives being questioned in an interview.

## What's real

| Component | Status |
|---|---|
| RDKit descriptor computation (MolWt, LogP, TPSA, QED, etc.) | Real |
| 2D structure rendering | Real |
| Solubility regression (RF + GB ensemble) | Real, trained on ESOL/Delaney (1128 molecules) |
| Toxicity classification (RF + GB ensemble) | Real, trained on Tox21 NR-AR assay (~7800 molecules) |
| Model-disagreement confidence | Real (RF vs GB spread, not simulated) |
| Lipinski / QED drug-likeness | Real (RDKit) |
| Batch virtual screening funnel | Real filtering at each stage |
| Descriptor-importance explanation | Real (RF feature importances) |

Current model performance (see `/api/health` or the in-app "Model Cards" tab):
- Solubility: ensemble R² ≈ 0.88, RMSE ≈ 0.76 log units, holdout of 226 molecules
- Toxicity (NR-AR): ensemble ROC-AUC ≈ 0.77, holdout of 1487 molecules

## What's explicitly out of scope (roadmap, not vaporware)

- Protein–ligand docking / 3D binding pose prediction — needs a docking
  engine (AutoDock Vina / Glide) and a target structure library
- Target-specific activity prediction via GNN — needs BindingDB/ChEMBL-scale
  bioactivity data and GPU training time
- Biomedical knowledge graph + literature RAG — needs a curated graph
  database and a document ingestion pipeline
- Molecular generation
- Multi-tenant auth, billing, job queues, cloud infra

These are the natural next milestones and are broken out that way on
purpose — a project that's honest about its boundary reads as more
senior than one that claims everything works.

## Architecture

```
Browser (vanilla JS, static/) 
      │
      ▼
FastAPI (main.py)
      │
      ├── RDKit  → descriptors, 2D rendering
      ├── joblib → trained sklearn models (solubility, toxicity)
      └── in-memory screening funnel over uploaded files
```

Single process, single container — intentionally simple to deploy and
to reason about in an interview.

## Run locally

```bash
cd backend
pip install -r requirements.txt
python train_models.py     # retrains both models from data/, ~30s
uvicorn main:app --reload
# open http://localhost:8000
```

## Deploy (pick one, ~5 minutes)

**Docker (any host):**
```bash
cd backend
docker build -t drugai-lite .
docker run -p 8000:8000 drugai-lite
```

**Render / Railway / Fly.io:** point them at `backend/` with the
included `Dockerfile` — no extra config needed. Expose port 8000.

**Hugging Face Spaces (Docker SDK):** push `backend/` as the repo root;
HF will build the Dockerfile automatically.

## Retraining

`train_models.py` downloads nothing at runtime — the datasets are
already in `data/`. Swap in your own CSV (needs a `smiles` column and
a target column) and adjust `train_models.py` to point at it if you
want to extend beyond solubility/toxicity.

## Project structure

```
backend/
  main.py              FastAPI app + inference logic
  train_models.py      Model training script (reproducible)
  data/                ESOL + Tox21 datasets
  models/              Trained model artifacts + metrics.json
  static/              Frontend (index.html, style.css, app.js)
  requirements.txt
  Dockerfile
```
