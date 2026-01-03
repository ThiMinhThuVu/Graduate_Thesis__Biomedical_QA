from utils.util import load_json, save_json
import os

def extract_with_models(query, extractor_model, target_labels):
    entities_set = set()
    
    doc = extractor_model(query)
    for ent in doc.ents:
        if ent.label_ in target_labels:
            entities_set.add(ent.text)

    return list(entities_set)

def extract_entities_from_evidence(extractor_model, retrieval_information, target_label, cache_filepath_retrieval, settings):
    output = load_json(cache_filepath_retrieval) if os.path.exists(cache_filepath_retrieval) else {}
    if 'ner' in output:
        output_ner = output['ner']
    else:
        output_ner = {}
        
    if settings['run'] in output_ner:
        list_entities = output_ner[settings['run']]
    else:
        print(f"Extracting entities.")
        list_entities = extract_with_models(retrieval_information, extractor_model, target_label)
        output_ner[settings['run']] = list_entities
        output['ner'] = output_ner
        save_json(output, cache_filepath_retrieval)
       
    all_key_entities = f"{target_label} Entities: " + ", ".join(list_entities)
    
    return all_key_entities

