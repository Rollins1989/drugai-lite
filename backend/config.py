"""Centralized DrugAI Lite configuration."""
from pathlib import Path
import os
BASE_DIR=Path(__file__).resolve().parent
MODELS_DIR=BASE_DIR/"models"; DATA_DIR=BASE_DIR/"data"; STATIC_DIR=BASE_DIR/"static"
APP_VERSION="4.0.0"
MAX_BATCH_MOLECULES=int(os.getenv("DRUGAI_MAX_BATCH_MOLECULES","5000"))
MAX_UPLOAD_BYTES=int(os.getenv("DRUGAI_MAX_UPLOAD_BYTES","10000000"))
REQUEST_TIMEOUT_SECONDS=int(os.getenv("DRUGAI_REQUEST_TIMEOUT_SECONDS","60"))
DESCRIPTOR_NAMES=["MolWt","LogP","TPSA","NumHDonors","NumHAcceptors","NumRotatableBonds","NumAromaticRings","RingCount","FractionCSP3","NumHeteroatoms","QED","NumValenceElectrons"]
