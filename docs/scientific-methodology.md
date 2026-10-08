# DrugAI Lite scientific methodology

## Scope

DrugAI Lite is a hypothesis-generation and computational prioritization workspace. It is not a clinical, regulatory, efficacy, safety, or experimental-validation system.

## Molecular descriptors

The bundled property models use a fixed 12-feature descriptor vector:

- Molecular weight
- Crippen LogP
- TPSA
- H-bond donors/acceptors
- Rotatable bonds
- Aromatic rings
- Ring count
- Fraction CSP3
- Heteroatoms
- QED
- Valence electrons

Descriptors are standardized with the fitted training scaler before model inference.

## Solubility model

The ESOL/Delaney task is treated as regression. The deployed prediction is the mean of a Random Forest regressor and Gradient Boosting regressor.

Primary metrics are R², RMSE and MAE. Random and Bemis–Murcko scaffold splits are supported by the training pipeline.

## Tox21 NR-AR model

The Tox21 NR-AR task is treated as binary classification. The deployed prediction is the mean probability from Random Forest and Gradient Boosting classifiers.

ROC-AUC and PR-AUC are the preferred discrimination metrics. Accuracy is retained as a secondary metric and should not be interpreted in isolation because class balance affects it.

## Target activity

Target models are optional. A target appears in the runtime catalog only when its required trained artifacts are present.

The included training pipeline can retrieve ChEMBL activity data, convert IC50 measurements to pIC50, use Morgan fingerprints, perform scaffold-aware evaluation, and train an RF/GB ensemble. Assay heterogeneity and dataset curation remain important limitations.

## Chemical space

Chemical-space analysis uses Morgan fingerprints followed by PCA and agglomerative clustering. The visualization is exploratory: PCA coordinates are not biological axes, and cluster membership is not a validated chemical-series classification.

## Applicability-domain signal

Nearest-neighbour Tanimoto similarity is used as a chemical-space proximity diagnostic. It is not a formal applicability-domain guarantee and does not establish prediction correctness.

## Sensitivity diagnostics

Local sensitivity perturbs one descriptor at a time by ±1 training standard deviation while holding other descriptors fixed. Predictions are passed through the fitted feature scaler.

This is a directional model diagnostic, not a causal explanation. It should not be interpreted as evidence that changing a molecular descriptor alone would produce the predicted biological outcome.

## Virtual screening

The screening score is a hand-weighted prioritization heuristic combining model outputs and rule-based signals. Pareto ranking provides a separate multi-objective view and should be preferred when users need to inspect trade-offs between objectives.

## Validation hierarchy

For serious scientific use, validation should progress from:

1. Random benchmark split
2. Scaffold-disjoint split
3. Temporal split when timestamps are available
4. Independent external dataset
5. Prospective experimental validation

No level of internal benchmark performance substitutes for independent or prospective validation.
