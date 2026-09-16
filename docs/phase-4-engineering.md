# Phase 4 — Engineering Architecture

Phase 4 turns the original monolithic FastAPI implementation into a layered application that is easier to test, review, extend, and deploy.

## Architecture

```text
backend/
├── api/
│   ├── __init__.py
│   └── routes.py          # HTTP contracts and status handling
├── chemistry.py           # RDKit parsing, descriptors, fingerprints, filters
├── config.py              # paths, versions, batch limits, feature configuration
├── model_service.py       # model loading, predictions, evaluation metadata
├── services.py            # analysis and virtual-screening orchestration
├── schemas.py             # Pydantic request validation
├── main.py                # application assembly only
├── train_models.py        # reproducible model training/evaluation
├── train_target_activity.py
└── tests/
    ├── test_api.py
    └── test_services.py
```

## Design goals

### Separation of concerns
- `main.py` only creates the FastAPI application, mounts the router, and serves the frontend.
- `api/routes.py` owns HTTP-level behavior.
- `schemas.py` owns request validation.
- `services.py` coordinates end-to-end application workflows.
- `chemistry.py` owns RDKit operations.
- `model_service.py` owns artifact loading and model inference.
- `config.py` centralizes filesystem paths and runtime limits.

### Model artifact visibility
The API now exposes both application and model versions through `/api/version` and `/api/health`. This makes it possible to distinguish a code release from the model artifacts used for inference.

If `backend/models/model_version.json` is absent, the API reports `unknown` rather than inventing a model version. Training in `train_models.py` creates this metadata when a new model bundle is produced.

### Runtime safety
- Batch screening remains capped at 5,000 molecules.
- Uploads are capped at 10 MB at the HTTP layer.
- Unsupported extensions are rejected before parsing.
- Required model artifacts fail fast during application startup instead of producing partial predictions.
- Docker now includes a healthcheck.
- CI runs dependency checks, Python compilation, tests, a Docker build, and a container health smoke test.

## Testing strategy

The test suite covers both external API contracts and internal service behavior:

- API availability and health/version contracts
- model-card metadata
- valid and invalid SMILES
- Pydantic validation
- scientific output naming (`model_agreement`, `screening_score`)
- screening score bounds and weight consistency
- CSV and SMI parsing
- small-batch screening integration
- target-model error handling
- deterministic descriptor calculation
- chemical-space signal thresholds

## Important scientific boundary
Phase 4 is an engineering improvement, not a scientific validation claim. Refactoring, stronger tests, version reporting, and Docker checks do not establish prospective accuracy, clinical validity, or safety. The existing model-evaluation methodology and limitations remain applicable.
