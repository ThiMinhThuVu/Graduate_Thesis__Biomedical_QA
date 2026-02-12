import json
import os
import csv
from postprocessing import postprocessing

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_mapping_csv(path):
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Giả sử có cột 'short_answer' và 'mapping' trong csv
            mapping[row['short_answer'].lower()] = row['mapping']
    return mapping

if __name__ == "__main__":
    cache_golden = load_json('data/all_answer_golden_61.json')
    metadata = load_json('data/metadata_medhop.json')
    mapping_data = load_json('wiki_heading_mapping.json')
    

    output_metadata = {}
    for qidx, item in metadata.items():
        list_golden = cache_golden[qidx]['True']
        all_cand = []
        
        for src in ["medhop_results_mcq_2_TFIDF1000/all_ans",
                    "medhop_results_mcq_3_TFIDF300/all_ans",
                    "medhop_results_no_mcq_TFIDF300/all_ans"]:

            if src == "medhop_results_mcq_2_TFIDF1000/all_ans":
                family = 'tfidf1000'
            elif src == "medhop_results_mcq_3_TFIDF300/all_ans":
                family = 'tfidf300_mcq'
            else:
                family = 'create_mcq'

            filepath = f"{src}/R4_step_3b_all_ans_{qidx}.json"
            if os.path.exists(filepath):
                sample = load_json(filepath)

                if family == 'create_mcq':
                    list_cand = sample.get('create_mcq', [])
                    for cand in list_cand:
                        key = cand.get('short_answer', '').lower()
                        norm_key = postprocessing(key)
                        
                        key_mapp = mapping_data.get(key, None)
                        if key_mapp is None:
                            key_mapp = mapping_data.get(norm_key, None)
                        cand['norm_ans'] = norm_key
                        cand['mapping'] = key_mapp
                        cand['source'] = 'create_mcq'
                        
                        if key in list_golden or norm_key in list_golden or key_mapp in list_golden:
                            cand['is_correct'] = True
                        else:
                            cand['is_correct'] = False
                            
                        all_cand.append(cand)

                    dict_src_cand = sample.get('1', {}).get('0', {}).get('0', {}).get('source_output', {})
                    for source, src_cand in dict_src_cand.items():
                        tmp_cand = src_cand.get('candidate', [{}])[0]
                        key = tmp_cand.get('short_answer', '').lower()
                        norm_key = postprocessing(key)

                        key_mapp = mapping_data.get(key, None)
                        if key_mapp is None:
                            key_mapp = mapping_data.get(norm_key, None)

                        tmp_cand['norm_ans'] = norm_key
                        tmp_cand['mapping'] = key_mapp
                        tmp_cand['source'] = source
                        if key in list_golden or norm_key in list_golden or key_mapp in list_golden:
                            tmp_cand['is_correct'] = True
                        else:
                            tmp_cand['is_correct'] = False
                            
                        if 'full_data' in tmp_cand:
                            del tmp_cand['full_data']
                        all_cand.append(tmp_cand)
                else:
                    # Xử lý cho tfidf1000, tfidf300_mcq nếu cần
                    pass
        
        if item['type_q'] == 'direct' and item['target_answer_type'] != 'correctness':
            output_metadata[qidx] = {
                'target_answer_type': item['target_answer_type'],
                'golden': list_golden,
                'candidates': all_cand
            }

    save_json(output_metadata, 'output_metadata_mapping.json')
                                            
                
