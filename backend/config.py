"""Application configuration and model paths for DrugAI Lite."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
APP_VERSION = "3.0.0"
MAX_BATCH_MOLECULES = 5000

DESCRIPTOR_NAMES = [
    "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "RingCount",
    "FractionCSP3", "NumHeteroatoms", "QED", "NumValenceElectrons",
]
