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
    import spacy
    extractor_model = spacy.load("en_ner_bc5cdr_md")
    extractor_model_2 = spacy.load("en_ner_jnlpba_md")
    extractor_model_3 = spacy.load("en_ner_bionlp13cg_md")
    from tqdm import tqdm

    run = 1
    output = {}
    # fileoutput = f'data/ner/SEQ/spacy_entity_9000q_{run}.json'
    # data = load_json(r'data\metadata_9000q.json')
    data = load_json(r'data\metadata_entity.json')

    # for i, (qidx, original_question) in enumerate(tqdm(data.items(), desc="LLM Extracting NER")):
        
    #     # Dừng hẳn khi đã xử lý xong 500 câu để tiết kiệm thời gian
    #     if i >= 500:
    #         break
        
    #     # Thực hiện trích xuất thực thể
    #     ext_entities = extract_ner_original_q_with_spacy(original_question)
    #     output[qidx] = {
    #         'id': qidx,
    #         'question': original_question,
    #         'extracted_entities': ext_entities
    #     }

    #     # Cứ mỗi 100 câu thì lưu một lần (i+1 vì i bắt đầu từ 0)
    #     if (i + 1) % 100 == 0:
    #         save_json(output, fileoutput)
    #         # Dòng print này giúp bạn theo dõi tiến độ trong console
    #         print(f"\n[Checkpoint] Đã lưu {i + 1} câu vào {fileoutput}")

    # # Lưu lần cuối để tránh sót những câu lẻ ở cuối (ví dụ câu thứ 3001-3005)
    # save_json(output, fileoutput)
    # print(f'\nFinished! Tổng số câu đã lưu: {len(output.keys())}')
    # print('Đường dẫn file: ', fileoutput)
        
    # seq_ner = load_json(r'data\ner\entity_extraction_1000_seq.json')
    
    
    # output = {}
    # all_ner = set() # Dùng set để lưu các thực thể duy nhất
    # BASE = "medhop_preprocessing/step1/1"
    # entity2wiki_path = r'data\cache\mapping\entity2wiki.json'
    
    # # 1. Duyệt qua dữ liệu để trích xuất NER
    # for qidx, item in tqdm(data.items(), desc="Extracting NER with SpaCy"):
    #     if item.get('type_q') != 'sequence':
    #         continue
        
    #     file_seq = os.path.join(BASE, f"step_1_{qidx}.json")
        
    #     if not os.path.exists(file_seq):
    #         continue
            
    #     sample = load_json(file_seq)
        
    #     # Duyệt qua các bước (steps) trong mỗi câu hỏi sequence
    #     steps_dict = sample.get('0', {})
    #     for step, step_data in steps_dict.items():
    #         ID = f"{qidx}|{step}"
    #         subquestion = step_data.get('subquestion', '')
            
    #         if not subquestion:
    #             continue

    #         # Trích xuất NER bằng hàm SpaCy của bạn
    #         ext_entities = extract_ner_original_q_with_spacy(extractor_model, extractor_model_2, extractor_model_3, subquestion)
            
    #         output[ID] = {
    #             'id': ID,
    #             'question': subquestion,
    #             'extracted_entities': ext_entities
    #         }
            
    #         # Lấy danh sách keys từ ext_entities['detail']
    #         # Dùng .update() thay vì .add() để đưa toàn bộ list vào set
    #         if 'detail' in ext_entities:
    #             entities_found = list(ext_entities['detail'].keys())
    #             all_ner.update(entities_found)

    # # Lưu kết quả mapping chi tiết
    # save_json(output, fileoutput)
    # print(f'\n[1] Saved mapping to: {fileoutput}')
    # print(f'Total IDs processed: {len(output)}')
    
    # # 2. Xử lý danh sách thực thể chưa có URL (Wikipedia)
    # if os.path.exists(entity2wiki_path):
    #     all_url = load_json(entity2wiki_path)
    #     # Loại bỏ những thực thể đã tồn tại trong file mapping wiki
    #     new_entities = list(all_ner - set(all_url.keys()))
    # else:
    #     print(f"Warning: {entity2wiki_path} not found. Saving all discovered entities.")
    #     new_entities = list(all_ner)

    # print(f'[2] Total unique new entities to search: {len(new_entities)}')
    
    # search_list_path = 'data/ner/SEQ/spacy_entity_9000q_search.json'
    # save_json(new_entities, search_list_path)
    # print(f'Saved search list to: {search_list_path}')
    
    cnt = 0
    seq_ner = load_json(r'data\ner\DIRECT1000\meta_entity_1000q_seq.json')
    BASE = "medhop_preprocessing/step1/1"
    for i, (qidx, item) in enumerate(tqdm(data.items(), desc="Extracting NER")):
        if item['type_q'] != 'sequence':
            continue
        
        file_seq = f"{BASE}/step_1_{qidx}.json"
        sample = load_json(file_seq)
               
        for step, step_data in sample['0'].items():
            ID = f"{qidx}|{step}"
            
            list_ner = list(seq_ner[ID]["extracted_entities"]["detail"].keys())
            step_data['key_entity'] = list_ner
        save_json(sample, file_seq)
        print(file_seq)
    print(cnt)