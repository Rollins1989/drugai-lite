"""DrugAI Lite application entry point.

The application is intentionally thin: HTTP routes, domain services, chemistry,
and model loading are separated so each layer can be tested independently.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import router
from config import APP_VERSION, STATIC_DIR

app = FastAPI(
    title="DrugAI Lite",
    version=APP_VERSION,
    description="Explainable computational drug-discovery screening prototype",
)
app.include_router(router)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
