# Release process

DrugAI Lite uses semantic versioning for application releases.

## Before release

1. Run the complete test suite from `backend/`.
2. Run Ruff and `pip check`.
3. Build the Docker image and verify `/api/v1/health`.
4. Verify the model manifest and evaluation metadata.
5. Review `CHANGELOG.md`.
6. Confirm scientific claims in README/model cards match the bundled artifacts.

## Versioning

- Application/API changes increment the application version when the public contract changes.
- Model bundle changes should update the model bundle version independently when appropriate.
- Breaking API changes belong under a new `/api/vN` namespace.

## Deployment boundary

The built-in asynchronous job store and custom reference library are process-local. Public multi-worker deployment requires a shared datastore/job queue, authentication, rate limiting and persistent job history before being treated as production infrastructure.
