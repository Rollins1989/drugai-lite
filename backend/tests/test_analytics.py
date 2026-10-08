from analytics import dataset_quality, pairwise_similarity, pareto_front, chemical_space

def test_pareto_front():
    rows=[{"activity":8,"toxicity_probability":.1,"qed":.8,"mw":300},{"activity":7,"toxicity_probability":.2,"qed":.7,"mw":320},{"activity":8.5,"toxicity_probability":.3,"qed":.9,"mw":340}]
    front=pareto_front(rows,["activity","toxicity","qed","mw"])
    assert rows[0] in front and rows[2] in front

def test_similarity_and_space():
    assert pairwise_similarity("CCO","CCCO")["tanimoto"] > 0
    result=chemical_space(["CCO","CCCO","c1ccccc1"])
    assert len(result["points"])==3

def test_dataset_quality():
    q=dataset_quality([{"smiles":"CCO"},{"smiles":"CCO"},{"smiles":"bad"}])
    assert q["valid_structures"]==2
    assert q["duplicate_structures"]==1
    assert q["invalid_structures"]==1
