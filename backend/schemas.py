"""Pydantic request schemas for DrugAI Lite API endpoints."""
from pydantic import BaseModel, Field


class MoleculeRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=5000)


class ActivityRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=5000)
    target: str = Field(default="egfr", min_length=1, max_length=100)
