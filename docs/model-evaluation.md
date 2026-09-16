# Model Evaluation & Scientific Interpretation

DrugAI Lite is a research/portfolio prototype. The evaluation is designed to demonstrate sensible validation practice, not to establish clinical, regulatory, or prospective performance.

## Validation design

### Solubility and Tox21

The bundled solubility and Tox21 models are evaluated on held-out data using the repository's existing random/scaffold evaluation artifacts. The exact reported metrics are stored in `backend/models/metrics.json`.

### EGFR activity

The target-specific pipeline uses:

1. ChEMBL activity records filtered to human targets, IC50 measurements with exact standard relation/units, valid pChEMBL values, and no data-validity comment.
2. One label per canonical SMILES using median aggregation across measurements.
3. Morgan fingerprints (radius 2, 2048 bits).
4. A Bemis-Murcko scaffold-disjoint holdout split.
5. Random Forest and Gradient Boosting regressors whose predictions are averaged for the ensemble.
6. A simple training-set mean baseline, reported alongside model metrics in future retraining runs.

The target training script records the retrieval timestamp, query parameters, dataset SHA-256, split seed, requested/actual test fraction, fingerprint configuration, and aggregation rule. This makes future ChEMBL refreshes auditable rather than silently treating a live API response as a fixed dataset.

## Why scaffold splitting?

Random splits can place close structural analogues in both training and test sets. A scaffold split keeps Bemis-Murcko scaffolds disjoint between the two partitions, providing a stricter test of chemical-space generalization.

The split is group-based, so the exact test fraction may differ from the requested 20% because an entire scaffold group must remain together. The training script records the actual fraction.

## Model agreement is not confidence

The API reports the absolute difference between the Random Forest and Gradient Boosting predictions as `model_agreement.absolute_disagreement`.

This is an ensemble disagreement heuristic. It is **not** a calibrated probability that a prediction is correct and should not be described as statistical confidence.

## Applicability-domain signal

The API reports the maximum Morgan/Tanimoto similarity to the bundled ESOL/Tox21 reference library. This is a chemical-space proximity signal. It is not a formal statistical applicability-domain method and does not guarantee prediction reliability.

## Explanations

The current explanation endpoint exposes tree-model `feature_importances_`. These are global model-level importance values. They should not be interpreted as causal explanations for an individual molecule.

## Screening score

The `screening_score` is a hand-weighted heuristic combining:

- modeled Tox21 NR-AR probability
- categorical solubility label
- QED
- reference-space proximity
- PAINS status
- Lipinski/Veber rule signals

The API exposes the weights and component contributions. The score is intended only for portfolio demonstration and candidate prioritization within this prototype. It is **not** a validated efficacy, safety, developability, or clinical score.

## Important limitations

- No external prospective validation is claimed.
- Tox21 NR-AR represents one assay endpoint, not overall human toxicity.
- ChEMBL activity data can contain assay and experimental heterogeneity.
- A model can perform well on a held-out benchmark while still failing on molecules outside its training distribution.
- Experimental testing is required before any biological or medicinal-chemistry conclusion.
