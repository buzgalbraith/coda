import os
import subprocess

import pandas as pd
from sentence_transformers import CrossEncoder

from coda.grounding.rag_grounder import RagGrounder



retrieval_model = RagGrounder()._pipeline.retriever
cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
STRUCTURED = True ## if to format queries to cross encoder in a structured way 

def load_dataset()->pd.DataFrame:
    """Loads in the MedCodEX benchmark from https://zenodo.org/records/13308316?preview_file=Readme.md"""
    os.makedirs("./MedCodEX", exist_ok=True)
    if not os.path.isfile("./MedCodEX/diagnosis.csv"):
        subprocess.call(['wget', '-P', './MedCodEX', 'https://zenodo.org/records/13308316/files/diagnosis.csv'])
    if not os.path.isfile("./MedCodEX/supporting_evidence.csv"):
        subprocess.call(['wget', '-P', './MedCodEX', 'https://zenodo.org/records/13308316/files/supporting_evidence.csv'])
    diagnosis = pd.read_csv("./MedCodEX/diagnosis.csv")
    supporting_evidence = pd.read_csv("./MedCodEX/supporting_evidence.csv").rename(columns={"ICD-10-cm code": "ICD10"})

    diag_with_evidence = diagnosis.merge(
        supporting_evidence,
        on = ['Document ID', "ICD10"],
        how='left'
    )
    return diag_with_evidence.groupby(
        ['Document ID', 'Diagnosis', 'ICD10']
    )["Supporting Evidence Text"].apply(list).reset_index()

def has_evidence(x):
    return not (len(x) == 1 and pd.isna(x[0]))

if __name__ == "__main__":
    benchmark = load_dataset()
    results = {
        'retrieved_ids' : [],
        'retrieval_scores' : [],
        'reranked_ids' : [],
        'reranked_scores' : []
    }
    for _, row in benchmark.iterrows():
        ## process row ## 
        diagnosis = row.get("Diagnosis")
        supporting_evidences = row.get("Supporting Evidence Text")
        supporting_evidence_text = "\n".join(supporting_evidences)if has_evidence(supporting_evidences) else ""
        retrieval_text = f"{diagnosis}\n\n{supporting_evidence_text}" if supporting_evidence_text else diagnosis
        ## retrieve candidates ## 
        retrieved_terms = retrieval_model.retrieve(
            retrieval_text,
        )
        ## re-rank with cross encoder and sort list ##  
        term_list = [x[0] for x in retrieved_terms]
        if STRUCTURED:
            pairs = [
                (f'Concept:{diagnosis}, Evidence:{supporting_evidence_text}',
                f"Identifier: {term.id}, Name: {term.name}")
                for term in term_list
            ]
        else:
            pairs = [
                (f'{diagnosis}, {supporting_evidence_text}',
                f"{term.id}, {term.name}")
                for term in term_list
            ]  
        scores = cross_encoder_model.predict(pairs)
        re_ranked_terms = sorted(zip(scores, term_list), reverse=True)
        ## save the results ## 
        results["retrieved_ids"].append([x[0].id.removeprefix("icd10:") for x in retrieved_terms])
        results["retrieval_scores"].append([x[1] for x in retrieved_terms])
        results['reranked_ids'].append([x[1].id.removeprefix("icd10:") for x in re_ranked_terms])
        results['reranked_scores'].append([x[0] for x in re_ranked_terms])
    ## Append results to date frame ## 
    benchmark['retrieved_ids'] = results["retrieved_ids"]
    benchmark['retrieval_scores'] = results['retrieval_scores']
    benchmark['reranked_ids'] = results["reranked_ids"]
    benchmark['reranked_scores'] = results['reranked_scores']
    ## evaluate ## 
    benchmark['icd10_retrieved'] = benchmark.apply(lambda r: r["ICD10"] in r["retrieved_ids"], axis=1)
    benchmark['reranked-hits@1'] = benchmark.apply(lambda r: r["ICD10"] in r["reranked_ids"][:1], axis=1)
    benchmark['reranked-hits@2'] = benchmark.apply(lambda r: r["ICD10"] in r["reranked_ids"][:2], axis=1)
    benchmark['reranked-hits@5'] = benchmark.apply(lambda r: r["ICD10"] in r["reranked_ids"][:5], axis=1)
    benchmark['reranked-hits@10'] = benchmark.apply(lambda r: r["ICD10"] in r["reranked_ids"], axis=1) # this will be equivalent to is retrieved but keeping for completeness
    ## display results ## 
    print("-"*30)
    print('Results:')
    print("Retrieval rate: {:.4f}".format(benchmark['icd10_retrieved'].sum()/len(benchmark)))
    print("Re-ranked: hits@1 {:.4f}".format(benchmark['reranked-hits@1'].sum()/len(benchmark)))
    print("Re-ranked: hits@2 {:.4f}".format(benchmark['reranked-hits@2'].sum()/len(benchmark)))
    print("Re-ranked: hits@5 {:.4f}".format(benchmark['reranked-hits@5'].sum()/len(benchmark)))
    print("Re-ranked: hits@10 {:.4f}".format(benchmark['reranked-hits@10'].sum()/len(benchmark)))