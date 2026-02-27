import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.candidate_cluster_ct6_no_zeroshot import process_get_final_answer
from postprocessing import postprocessing
from sentence_transformers import SentenceTransformer
embedding_model_cluster = SentenceTransformer(
            "pritamdeka/S-BioBert-snli-multinli-stsb",
            device='cpu'
        )

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
BASE = "data/ALL_ROUND/1000/medhop_results_gg_direct/all_ans"
golden = load_json(r'data\all_answer_golden_56_update.json')
metadata = load_json(r'data\metadata_medhop.json')

PATH_MAPPING = r'data\mapping_answer_gol_add_seq.json'
mapping_data = load_json(PATH_MAPPING)
o3_cache_zeroshot = load_json(r'data\cache\O3_zeroshot_ans_cache.json')
dict_output = {}

cnt = 0
cnt_true = 0
list_qidx = []
for idx, (qidx, item) in enumerate(metadata.items()):    
    if item['type_q'] != 'direct' or item["target_answer_type"] != 'correctness' or golden[qidx]['True'] == []:
        continue
    
    file_path = f"{BASE}/R4_step_3b_all_ans_{qidx}.json"
    
    data = load_json(file_path)
        
    cluster_genered_answer = data['1']['0']['0']['cluster_genered_answer']
    o3_cache_file = r'data\cache\O3_zeroshot_ans_cache_direct.json'
   
    cnt += 1    
    final_answer = process_get_final_answer(qidx, "", "1", "0", "0", "WH", cluster_genered_answer, mapping_data, embedding_model_cluster, o3_cache_file)  
    # dict_output[qidx] = final_answer
    # final_answer = data[]
    status = final_answer['selection_status']
    
    if status == 'selected':
        data['1']['0']['0']['final_answer'] = final_answer
    else:
        data['1']['0']['0']['final_answer'] = {
            'short_answer': o3_cache_zeroshot[qidx]['short_answer'],
            'long_answer': o3_cache_zeroshot[qidx]['long_answer'],
            'selection_status': "zero-shot_fallback",
            "cluster_score": None
        }

    data['final_answer'] = {
        'short_answer': data['1']['0']['0']['final_answer']['short_answer'],
        'long_answer': data['1']['0']['0']['final_answer']['long_answer'] 
    }
    
    save_json(data, file_path)
    # print(file_path)