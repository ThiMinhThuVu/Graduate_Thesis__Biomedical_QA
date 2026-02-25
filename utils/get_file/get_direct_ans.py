import json
import  os

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
metadata = load_json('data/metadata_9000q.json')
BASE = "DIRECT_9000_medhop_results/all_ans"
cnt = 0
for qidx, item in metadata.items():
    if item['type_q'] != 'direct':
        continue
    
    file_path = f"{BASE}/R4_step_3b_all_ans_{qidx}.json"
    if os.path.exists(file_path):       
        data = load_json(file_path)
        final_answer = data['1']['0']['0']['final_answer']
        
        if final_answer['selection_status'] == 'selected' and final_answer['cluster_score'] < 2:
            data['1']['0']['0']['final_answer'] = {
                                                    "short_answer": "HELLO THU",
                                                    "long_answer": "HELLO THU",
                                                    "selection_status": "zero-shot_fallback",
                                                    "cluster_score": None
                                                }
            data['1']['final_qi']['short_answer'] = "HELLO THU"
            data['1']['final_qi']['long_answer'] = "HELLO THU"
            data['final_answer'] = {
                "long_answer": "HELLO THU",
                "short_answer": "HELLO THU"
            }
            cnt += 1
            print(file_path)
            save_json(data, file_path)
        
print(cnt)
    