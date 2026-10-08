# Changelog

All notable changes to DrugAI Lite are documented here.

## 4.0.0

### Platform
- FastAPI computational drug-discovery workspace with browser UI.
- Versioned model/runtime provenance.
- Molecular analysis, comparison, virtual screening, chemical-space analysis and Pareto ranking.
- Docker healthcheck, CI, CodeQL and Dependabot.

### Scientific
- ESOL solubility and Tox21 NR-AR ensemble models.
- Random and Bemis–Murcko scaffold evaluation pipeline.
- Empirical model-agreement and RF-tree spread diagnostics.
- Local one-feature sensitivity diagnostics.
- Optional target-specific ChEMBL training pipeline.

### P4 maturity
- Versioned `/api/v1` API surface with backward-compatible `/api` routes.
- Prometheus-compatible request counters and latency metrics.
- First-class evaluation metadata endpoint.
- Reproducibility/evaluation documentation.
- Runtime dependency separation and non-root container execution.
