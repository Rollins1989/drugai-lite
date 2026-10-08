"""Pydantic API contracts."""
from pydantic import BaseModel, Field
class MoleculeRequest(BaseModel): smiles:str=Field(min_length=1,max_length=5000)
class CompareRequest(BaseModel): smiles_a:str=Field(min_length=1,max_length=5000); smiles_b:str=Field(min_length=1,max_length=5000)
class ActivityRequest(BaseModel): smiles:str=Field(min_length=1,max_length=5000); target:str=Field(default="egfr",min_length=1,max_length=100)
class ParetoRequest(BaseModel): rows:list[dict]; objectives:list[str]=Field(default=["activity","toxicity","solubility","qed"])
class ChemicalSpaceRequest(BaseModel): smiles:list[str]=Field(min_length=2,max_length=5000); labels:list[str]|None=None; n_clusters:int=Field(default=6,ge=2,le=20)
