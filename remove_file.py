import os
import json
import shutil
# threads = [500,1000]
threads = [0, 500]

dir1 = "medhop_results_mcq_2_TFIDF1000/all_ans"
dir2 = "medhop_results/all_ans"


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

metadata = load_json("data/metadata_medhop.json")
raw_data = load_json(r'data\important_id.json')

cnt = 0
for idx, (qidx, sample_raw) in enumerate(raw_data.items()):
    if not (threads[0] <= idx < threads[1]):
        continue
    
    sample = metadata[qidx]
    if sample['type_q'] != 'direct' or sample['target_answer_type'] == 'correctness':
        continue
    
    filepath = f"{dir1}/R4_step_3b_all_ans_{qidx}.json"
    if os.path.exists(filepath):
        # src_path = f"{dir2}/R4_step_3b_all_ans_{qidx}.json"
        # shutil.copy(filepath, src_path)
        cnt += 1
        
print(cnt)