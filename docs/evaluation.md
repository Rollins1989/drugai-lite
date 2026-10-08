# Evaluation and validation

DrugAI Lite separates **model evaluation evidence** from deployed prediction outputs.

## Reproducible evaluation

Run:

```bash
cd backend
python train_models.py
```

The training pipeline:

1. loads the pinned ESOL and Tox21 datasets;
2. records SHA-256 dataset hashes;
3. computes the descriptor feature matrix;
4. evaluates both random and Bemis–Murcko scaffold-disjoint splits;
5. reports regression and classification metrics against simple baselines;
6. writes ROC, precision-recall, calibration, observed-vs-predicted and residual plots;
7. trains the deployment bundle on the scaffold split;
8. writes model/evaluation metadata.

Generated evaluation outputs belong under `backend/models/evaluation/` and are intentionally treated as generated artifacts rather than source code.

## Metrics

For solubility, prioritize ensemble R², RMSE and MAE. For Tox21 NR-AR, prioritize ROC-AUC, PR-AUC, Brier score and calibration diagnostics. Accuracy alone is not an adequate characterization of a potentially imbalanced assay endpoint.

## Validation hierarchy

The project uses this hierarchy:

1. random benchmark split;
2. scaffold-disjoint split;
3. temporal split where timestamps are available;
4. independent external dataset;
5. prospective experimental validation.

Only the first two are currently automated by the bundled training pipeline.

## Interpretation boundary

Evaluation metrics describe performance on held-out benchmark data. They do not establish clinical utility, prospective efficacy, general human toxicity, or performance on an unrelated chemical distribution.

The deployed uncertainty fields are diagnostic signals. They are not calibrated confidence or prediction intervals.

## API contract

`GET /api/v1/evaluation` exposes the current evaluation metadata and validation hierarchy. The legacy `/api/evaluation` path remains available for compatibility.
