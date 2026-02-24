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
    # import spacy
     # extractor_model = spacy.load("en_ner_bc5cdr_md")
    # extractor_model_2 = spacy.load("en_ner_jnlpba_md")
    # extractor_model_3 = spacy.load("en_ner_bionlp13cg_md")
    from tqdm import tqdm

    run = 2
    output = {}
    fileoutput = f'data/ner/LLM/LLM_entity_9000q_{run}.json'
    data = load_json(r'data\qidx_question_9000.json')

    # Hoặc cứ dùng enumerate như cũ:
    for i, (qidx, original_question) in enumerate(tqdm(data.items(), desc="LLM Extracting NER")):
        
        # Dừng hẳn khi đã xử lý xong 3000 câu để tiết kiệm thời gian
        if i < 3000 or i > 6000:
            continue
        
        # Thực hiện trích xuất thực thể
        ext_entities = extract_bio_entity_from_original_question(original_question)
        output[qidx] = {
            'id': qidx,
            'question': original_question,
            'extracted_entities': ext_entities
        }

        # Cứ mỗi 100 câu thì lưu một lần (i+1 vì i bắt đầu từ 0)
        if (i + 1) % 100 == 0:
            save_json(output, fileoutput)
            # Dòng print này giúp bạn theo dõi tiến độ trong console
            print(f"\n[Checkpoint] Đã lưu {i + 1} câu vào {fileoutput}")

    # Lưu lần cuối để tránh sót những câu lẻ ở cuối (ví dụ câu thứ 3001-3005)
    save_json(output, fileoutput)
    print(f'\nFinished! Tổng số câu đã lưu: {len(output.keys())}')
    print('Đường dẫn file: ', fileoutput)

    # for i, (qidx, original_question) in enumerate(tqdm(data.items(), desc="Extracting NER")):
    #     # ext_entities = extract_ner_original_q_with_spacy(extractor_model, extractor_model_2, extractor_model_3, original_question)
    #     # output[qidx] = {
    #     #     'original_question': original_question,
    #     #     'entities_q': ext_entities
    #     # }
        
    #     if i > 3000:
    #         continue
        
    #     ext_entities = extract_bio_entity_from_original_question(original_question)
    #     output[qidx] = {
    #         'id': qidx,
    #         'question': original_question,
    #         'extracted_entities': ext_entities
    #     }

    # save_json(output, fileoutput)
    # print('save to file: ', fileoutput)
    # print(len(output.keys()))