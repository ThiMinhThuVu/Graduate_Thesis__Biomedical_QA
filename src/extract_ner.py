from utils.util import load_json, save_json
import os

def extract_with_models(query, extractor_model, target_labels):
    entities_set = set()
    
    doc = extractor_model(query)
    for ent in doc.ents:
        if ent.label_ in target_labels:
            entities_set.add(ent.text)

    return list(entities_set)

def extract_entities_from_evidence(extractor_model, key_entities, retrieval_information, target_label, step, paths):
    ner_filepath = paths['ner']
    output_ner = load_json(ner_filepath) if os.path.exists(ner_filepath) else {}
    
    if step in output_ner:
        list_entities = output_ner[step]
    else:
        print(f"Extracting entities.")
        list_entities = extract_with_models(retrieval_information, extractor_model, target_label)
        output_ner[step] = list_entities
        save_json(output_ner, ner_filepath)
   
    # all_key_entities = ""
    # for item in key_entities:
    #     all_key_entities += f"{item}, "
    
    all_key_entities = f"{target_label} Entities: " + ", ".join(list_entities)
    
    return all_key_entities

