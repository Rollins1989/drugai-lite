"""RDKit chemistry utilities used by the API and screening services."""
import base64
import csv
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator

from config import DATA_DIR, DESCRIPTOR_NAMES

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

_params = FilterCatalogParams()
_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
PAINS = FilterCatalog(_params)


def parse_mol(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def compute_descriptors(mol) -> dict:
    return {
        "MolWt": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Crippen.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": Lipinski.NumAromaticRings(mol),
        "RingCount": Descriptors.RingCount(mol),
        "FractionCSP3": round(Descriptors.FractionCSP3(mol), 3),
        "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),
        "QED": round(QED.qed(mol), 3),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "ExactMolWt": round(Descriptors.ExactMolWt(mol), 4),
        "NumRings": rdMolDescriptors.CalcNumRings(mol),
    }


def descriptor_vector(desc: dict):
    import numpy as np
    return np.array([[desc[n] for n in DESCRIPTOR_NAMES]], dtype=float)


def lipinski_verdict(desc: dict) -> dict:
    violations = []
    if desc["MolWt"] > 500: violations.append("Molecular weight > 500")
    if desc["LogP"] > 5: violations.append("LogP > 5")
    if desc["NumHDonors"] > 5: violations.append("H-bond donors > 5")
    if desc["NumHAcceptors"] > 10: violations.append("H-bond acceptors > 10")
    return {"pass": len(violations) <= 1, "violations": violations, "n_violations": len(violations)}


def veber_verdict(mol) -> dict:
    violations = []
    if Descriptors.NumRotatableBonds(mol) > 10: violations.append("Rotatable bonds > 10")
    if Descriptors.TPSA(mol) > 140: violations.append("TPSA > 140 Å²")
    return {"pass": not violations, "violations": violations}


def structural_alerts(mol) -> dict:
    names = []
    for match in PAINS.GetMatches(mol):
        try: names.append(match.GetDescription())
        except Exception: names.append("PAINS alert")
    names = list(dict.fromkeys(names))
    return {"pass": not names, "count": len(names), "alerts": names[:10]}


def molecular_identity(mol) -> dict:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return {
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "murcko_scaffold_smiles": Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else "",
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(mol),
    }


def fingerprint(mol):
    return MORGAN.GetFingerprint(mol)


def structure_svg_b64(mol) -> str:
    from rdkit.Chem.Draw import rdMolDraw2D
    drawer = rdMolDraw2D.MolDraw2DSVG(360, 280)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return base64.b64encode(drawer.GetDrawingText().encode()).decode()


def load_reference_library():
    refs = []
    for filename, label in [("delaney.csv", "ESOL"), ("tox21.csv", "Tox21")]:
        path = DATA_DIR / filename
        if not path.exists(): continue
        with path.open(newline="", encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                smi = (row.get("smiles") or "").strip()
                mol = parse_mol(smi)
                if mol is not None: refs.append((smi, label, fingerprint(mol)))
    return refs

REFERENCE_LIBRARY = load_reference_library()


def nearest_analogs(mol, limit=5):
    fp = fingerprint(mol)
    scored = [(DataStructs.TanimotoSimilarity(fp, ref_fp), smi, source) for smi, source, ref_fp in REFERENCE_LIBRARY]
    scored.sort(reverse=True)
    return [{"smiles": smi, "source": source, "tanimoto": round(score, 3)} for score, smi, source in scored[:limit]]
