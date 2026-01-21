from utils.util import load_json, save_json
import os

WIKI_LABELS_LIST = ["PERSON", "WORK_OF_ART", "GPE", "ORG", "LOC", "EVENT", "CARDINAL", "NORP", "LANGUAGE"]
BIO_LABLES_1 = ["DISEASE", "CHEMICAL"]
BIO_LABLES_2 = ["DNA", "CELL_TYPE", "CELL_LINE", "RNA", "PROTEIN"]
BIO_LABLES_3 = ["AMINO_ACID", "ANATOMICAL_SYSTEM", "CANCER", "CELL", "CELLULAR_COMPONENT", "DEVELOPING_ANATOMICAL_STRUCTURE", "GENE_OR_GENE_PRODUCT", "IMMATERIAL_ANATOMICAL_ENTITY", "MULTI-TISSUE_STRUCTURE", "ORGAN", "ORGANISM", "ORGANISM_SUBDIVISION", "ORGANISM_SUBSTANCE", "PATHOLOGICAL_FORMATION", "SIMPLE_CHEMICAL", "TISSUE"]

def extract_with_models(query, target_labels, extractor_model, extractor_model_2, extractor_model_3):
    entities_set = set()

    labels_wiki = [label for label in target_labels if label in WIKI_LABELS_LIST]
    labels_1 = [label for label in target_labels if label in BIO_LABLES_1]
    labels_2 = [label for label in target_labels if label in BIO_LABLES_2]
    labels_3 = [label for label in target_labels if label in BIO_LABLES_3]
    
    if labels_wiki:
        doc = extractor_model(query)
        for ent in doc.ents:
            if ent.label_ in labels_wiki:
                entities_set.add(ent.text)
        
    if labels_1:
        doc1 = extractor_model(query)
        for ent in doc1.ents:
            if ent.label_ in labels_1:
                entities_set.add(ent.text)

    if labels_2 and extractor_model_2:
        doc2 = extractor_model_2(query)
        for ent in doc2.ents:
            if ent.label_ in labels_2:
                entities_set.add(ent.text)

    if labels_3 and extractor_model_3:
        doc3 = extractor_model_3(query)
        for ent in doc3.ents:
            if ent.label_ in labels_3:
                entities_set.add(ent.text)

    return list(entities_set)

def extract_entities_from_evidence(extractor_model, extractor_model_2, extractor_model_3, retrieval_information, target_label, cache_filepath_retrieval, settings):
    output = load_json(cache_filepath_retrieval) if os.path.exists(cache_filepath_retrieval) else {}
    if 'ner' in output:
        output_ner = output['ner']
    else:
        output_ner = {}
        
    if settings['run'] in output_ner:
        list_entities = output_ner[settings['run']]
    else:
        print(f"Extracting entities.")
        list_entities = extract_with_models(retrieval_information, target_label, extractor_model, extractor_model_2, extractor_model_3)
        output_ner[settings['run']] = list_entities
        output['ner'] = output_ner
        save_json(output, cache_filepath_retrieval)
       
    all_key_entities = f"{target_label} Entities: " + ", ".join(list_entities)
    
    return all_key_entities

