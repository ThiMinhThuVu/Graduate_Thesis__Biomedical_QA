import sys
import json
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.util import load_json, save_json
from src.prompt import load_entity_extraction_prompt
from src.gpt_request import request_api

WIKI_LABELS_LIST = ["PERSON", "WORK_OF_ART", "GPE", "ORG", "LOC", "EVENT", "CARDINAL", "NORP", "LANGUAGE"]
# BIO_LABLES_1 = ["DISEASE", "CHEMICAL"]
# BIO_LABLES_2 = ["DNA", "CELL_TYPE", "CELL_LINE", "RNA", "PROTEIN"]
# BIO_LABLES_3 = ["AMINO_ACID", "ANATOMICAL_SYSTEM", "CANCER", "CELL", "CELLULAR_COMPONENT", "DEVELOPING_ANATOMICAL_STRUCTURE", "GENE_OR_GENE_PRODUCT", "IMMATERIAL_ANATOMICAL_ENTITY", "MULTI-TISSUE_STRUCTURE", "ORGAN", "ORGANISM", "ORGANISM_SUBDIVISION", "ORGANISM_SUBSTANCE", "PATHOLOGICAL_FORMATION", "SIMPLE_CHEMICAL", "TISSUE"]

#NER #1
BIO_LABLES_1 = ["DISEASE", "CHEMICAL"]
BIO_LABLES_2 = ["DNA", "CELL_TYPE", "CELL_LINE", "RNA", "PROTEIN"]
BIO_LABLES_3 = ["AMINO_ACID", "ANATOMICAL_SYSTEM", "CANCER", "CELL", "CELLULAR_COMPONENT", "GENE_OR_GENE_PRODUCT", "ORGAN", "ORGANISM", "SIMPLE_CHEMICAL", "TISSUE"]

def extract_with_models(query, target_labels, extractor_model, extractor_model_2, extractor_model_3):
    entities_set = set()
    lower_entites = set()

    # labels_wiki = [label for label in target_labels if label in WIKI_LABELS_LIST]
                    
    labels_1 = [label for label in target_labels if label in BIO_LABLES_1]
    labels_2 = [label for label in target_labels if label in BIO_LABLES_2]
    labels_3 = [label for label in target_labels if label in BIO_LABLES_3]
        
    if labels_1 != [] and extractor_model:
        print("[Extract NER] - Model 1: en_ner_bc5cdr_md")
        doc1 = extractor_model(query)
        for ent in doc1.ents:
            if ent.label_ in labels_1:
                ent_text_lower = ent.text.lower()
                if ent_text_lower not in lower_entites:
                    lower_entites.add(ent_text_lower)
                    entities_set.add(ent.text)
        
    if labels_2 != [] and extractor_model_2:
        print("[Extract NER] - Model 2: en_ner_jnlpba_md")
        doc2 = extractor_model_2(query)
        for ent in doc2.ents:
            if ent.label_ in labels_2:
                ent_text_lower = ent.text.lower()
                if ent_text_lower not in lower_entites:
                    lower_entites.add(ent_text_lower)
                    entities_set.add(ent.text)

    if labels_3 != [] and extractor_model_3:
        print("[Extract NER] - Model 3: en_ner_bionlp13cg_md")
        doc3 = extractor_model_3(query)
        for ent in doc3.ents:
            if ent.label_ in labels_3:
                ent_text_lower = ent.text.lower()
                if ent_text_lower not in lower_entites:
                    lower_entites.add(ent_text_lower)
                    entities_set.add(ent.text)

                 
    return list(entities_set)

def extract_entities_from_evidence(extractor_model, extractor_model_2, extractor_model_3, retrieval_information, target_label):   
    list_entities = extract_with_models(retrieval_information, target_label, extractor_model, extractor_model_2, extractor_model_3)
    all_key_entities = ", ".join(list_entities)
    return all_key_entities


def extract_bio_entity_from_original_question(question, model='gpt-4o-mini'):
    messages = load_entity_extraction_prompt(question=question)
    result_json = request_api(messages, model)
    try:
        result_dict = json.loads(result_json)
    
        if isinstance(result_dict, list):
            return result_dict
        else:
            return []
    except Exception as e:
        print(f"[extract_bio_entity_from_original_question] Error parsing API result: {e}")
        return []


def extract_ner_original_q_with_spacy(ext_model_1, ext_model_2, ext_model_3, question):
    # Trích xuất entity từ cả 3 model
    
    detail = {}
    cache = []
    lower_entities = set()
    models = [ext_model_1, ext_model_2, ext_model_3]
    bio_labels_list = [BIO_LABLES_1, BIO_LABLES_2, BIO_LABLES_3]
    for idx, (model, bio_labels) in enumerate(zip(models, bio_labels_list), 1):
        doc = model(question)
        for ent in doc.ents:
            ent_text = ent.text.strip()
            ent_text_lower = ent_text.lower()
            if ent_text_lower not in lower_entities:
                lower_entities.add(ent_text_lower)
                cache.append(ent_text)
                detail[ent_text] = {
                    'label': ent.label_,
                    'bio_type': 'bio' if ent.label_ in bio_labels else 'non-bio',
                    'source': [idx]
                }
            else:
                # Nếu entity đã có, chỉ bổ sung model vào source nếu chưa có
                if ent_text in detail:
                    if 'source' in detail[ent_text] and idx not in detail[ent_text]['source']:
                        detail[ent_text]['source'].append(idx)
    result = {'ner': cache,
              'detail': detail}
    return result


if __name__ == "__main__":
   
    from tqdm import tqdm

    run = 1
    output = {}
    fileoutput = f'data/ner/SEQ/LLM_entity_9000q_{run}.json'
    data = load_json(r'data\metadata_9000q.json')
    
    output_jsonl = r'src\batch_api\input\extract_entity_4omini.jsonl'
    
    BASE_DIR = "medhop_preprocessing/step1/1"
    with open(output_jsonl, "w", encoding="utf-8") as out:
        for qidx, item in tqdm(data.items(), desc="Processing Questions"):
            if item.get('type_q') != 'sequence':
                continue
            
            file_seq = os.path.join(BASE_DIR, f"step_1_{qidx}.json")
            
            if not os.path.exists(file_seq):
                # print(f"Warning: File {file_seq} not found. Skipping.")
                continue
                
            sample = load_json(file_seq)
            
            # Giả sử cấu trúc sample là {'0': { 'step1': {...}, 'step2': {...} }}
            steps_data = sample.get('0', {})
            
            for step, step_content in steps_data.items():
                ID = f"{qidx}|{step}"
                subquestion = step_content.get('subquestion', '')
                
                if not subquestion:
                    continue

                messages = load_entity_extraction_prompt(question=subquestion)
                
                # Cấu trúc record chuẩn cho Batch API
                record = {
                    "custom_id": ID, 
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": messages,
                        "response_format": { "type": "json_object" }, # Bắt buộc Prompt trả về { }
                        "max_tokens": 1000,
                        "temperature": 0 # Thường để 0 cho tác vụ NER để kết quả ổn định
                    }
                }

                # Ghi duy nhất 1 dòng cho mỗi record
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                output[ID] = {
                    'id': ID,
                    'question': subquestion
                }
    
    save_json(output, fileoutput)
    print(f'\n--- Hoàn tất ---')
    print(f'Đã lưu file JSONL: {output_jsonl}')
    print(f'Đã lưu file Mapping: {fileoutput}')
    print(f'Tổng số subquestions đã xử lý: {len(output)}')