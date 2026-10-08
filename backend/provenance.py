"""Runtime/model provenance and structured request metadata."""
from __future__ import annotations
import hashlib, platform
from datetime import datetime, timezone
import rdkit, sklearn

def sha256_text(value:str)->str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def runtime_provenance(app_version:str, model_version:dict)->dict:
    return {
        "application_version":app_version,
        "model_version":model_version,
        "python_version":platform.python_version(),
        "platform":platform.platform(),
        "rdkit_version":getattr(rdkit,"__version__","unknown"),
        "scikit_learn_version":getattr(sklearn,"__version__","unknown"),
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
    }
