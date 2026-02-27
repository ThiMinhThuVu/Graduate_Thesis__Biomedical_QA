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
        
BASE = "DIRECT_9000_medhop_results/all_ans"
golden = load_json(r'data\all_answer_golden_56_update.json')
metadata = load_json(r'data\metadata_9000q.json')

PATH_MAPPING = r'data\mapping_answer_gol_add_seq.json'
mapping_data = load_json(PATH_MAPPING)

dict_output = {}

cnt = 0
cnt_true = 0
list_qidx = []
for idx, (qidx, item) in enumerate(metadata.items()):    
    if item['type_q'] != 'direct':# or item["target_answer_type"] != 'correctness' or golden[qidx]['True'] == []:
        continue
    
    file_path = f"{BASE}/R4_step_3b_all_ans_{qidx}.json"
    
    data = load_json(file_path)
    if data['1']['0']['0']['target_answer_type'] == 'correctness':
        cnt += 1
    continue
    
    status = data['1']['0']['0']['final_answer']['selection_status']
    
    short_ans = data['1']['0']['0']['final_answer']['short_answer'].lower()
    
    cluster_genered_answer = data['1']['0']['0']['cluster_genered_answer']
    o3_cache_file = r'data\cache\O3_zeroshot_ans_cache_direct.json'
        
    if status == "selected":
        # continue  
        cnt += 1    
        final_answer = process_get_final_answer(qidx, "", "1", "0", "0", "WH", cluster_genered_answer, mapping_data, embedding_model_cluster, o3_cache_file)  
        dict_output[qidx] = final_answer
        if short_ans in golden[qidx]['True']:
            cnt_true += 1
            list_qidx.append(qidx)                  
    else:
        # continue
        dict_output[qidx] = data['1']['0']['0']['final_answer']
        cnt += 1   
        if short_ans in mapping_data:
            short_ans = mapping_data[short_ans]
        else:
            short_ans = postprocessing(short_ans)
            short_ans = mapping_data.get(short_ans, short_ans)            
       
        short_ans = postprocessing(short_ans)
        if short_ans in golden[qidx]['True']:
                cnt_true += 1
                list_qidx.append(qidx)    
            
print(cnt)
print(cnt_true)
# save_json(list_qidx, 'tmp_idx.json')
save_json(dict_output, r'src\cluster\final\YN_answer_CT6_#5.json')
print(len(dict_output.keys()))
print('src\cluster\final\YN_answer_CT6_#5.json')