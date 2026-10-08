"""Exportable research reports. Reports are computational summaries, not scientific validation."""
from __future__ import annotations
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def molecule_pdf(result:dict, provenance:dict)->bytes:
    buf=BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=letter,rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36)
    styles=getSampleStyleSheet(); story=[Paragraph("DrugAI Lite — Molecular Assessment",styles["Title"])]
    identity=result["identity"]; desc=result["descriptors"]
    story += [Paragraph("Computational research report — not clinical, regulatory, or experimental evidence.",styles["Normal"]),Spacer(1,12)]
    data=[["Field","Value"],["Formula",identity["formula"]],["Canonical SMILES",identity["canonical_smiles"]],["Murcko scaffold",identity["murcko_scaffold_smiles"] or "Acyclic"]]
    for k in ["MolWt","LogP","TPSA","NumHDonors","NumHAcceptors","NumRotatableBonds","QED","FractionCSP3"]:
        data.append([k,str(desc[k])])
    data += [["Solubility",str(result["solubility"]["log_solubility_mol_per_L"])],["Tox21 probability",str(result["toxicity"]["toxicity_probability"])],["Heuristic prioritization",str(result["screening_score"]["value"])],["Domain status",result["applicability_domain"]["status"]]]
    t=Table(data,colWidths=[180,340]); t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.4,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story += [t,Spacer(1,14),Paragraph("Model/provenance",styles["Heading2"]),Paragraph(str(provenance),styles["BodyText"]),Spacer(1,10),Paragraph(result["screening_score"]["note"],styles["BodyText"])]
    doc.build(story); return buf.getvalue()
