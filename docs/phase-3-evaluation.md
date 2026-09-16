# Phase 3 — ML Evaluation & Engineering

Phase 3 makes the training workflow more auditable and gives the portfolio project standard model-diagnostic artifacts instead of relying only on headline metrics.

## Evaluation outputs

Running `python train_models.py` from `backend/` now writes diagnostic plots to `backend/models/evaluation/`.

### Regression

For ESOL solubility, the training script generates:

- observed-vs-predicted scatter plots
- residual-distribution plots
- RF, Gradient Boosting and ensemble metrics
- a training-mean baseline

Metrics include R², RMSE and MAE. The baseline makes it possible to see whether the learned model provides useful predictive signal beyond a constant prediction.

### Classification

For Tox21 NR-AR, the training script generates:

- ROC curves
- precision-recall curves
- reliability/calibration diagrams
- RF, Gradient Boosting and ensemble metrics
- prevalence baselines

In addition to ROC-AUC, the workflow reports PR-AUC, precision, recall, F1 and Brier score. PR-AUC is particularly useful when the positive class is imbalanced.

The calibration plot is descriptive model diagnostics, not evidence that the ensemble probabilities are calibrated. No post-hoc calibration model is currently fitted.

## Reproducibility

Training records:

- UTC generation timestamp
- random seed
- requested test fraction
- actual test fraction for each split
- dataset SHA-256
- deployment split
- model version

The target-specific ChEMBL workflow additionally records retrieval/query metadata and the target dataset hash when it is retrained.

## Deployment model policy

The API deployment artifacts are trained using the scaffold split. Random-split results remain in the evaluation record for comparison, but the scaffold split is treated as the primary generalization check because it reduces scaffold overlap between training and held-out molecules.

## Engineering checks

CI now performs:

1. dependency installation with pip caching
2. Python source compilation
3. the API test suite

The tests cover endpoint contracts, request validation, scientifically named outputs, heuristic-score constraints, invalid SMILES handling, and batch-file validation.

## Current limitations

- The project does not claim prospective or external validation.
- The calibration diagram is diagnostic only; probabilities are not guaranteed to be calibrated.
- The ensemble contains two related tree-model families rather than independent model classes from substantially different inductive biases.
- `main.py` remains a single application module; a future engineering phase can separate chemistry utilities, prediction services, schemas, and API routes into dedicated packages.
- Model artifacts should be regenerated after changes to datasets, feature definitions, hyperparameters, or preprocessing.
