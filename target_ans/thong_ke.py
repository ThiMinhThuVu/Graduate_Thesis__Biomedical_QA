import json 
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_json(path):
    if not os.path.exists(path):
        print(f"Lỗi: Không tìm thấy file tại {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
folder = "medhop_preprocessing/step1/1"
metadata = load_json(r'data\metadata_medhop.json')

# Import process_batch from pipeline_target_answer_type_batch.py
from pipeline_target_answer_type_batch import process_batch

def process_sequence_questions():
    cnt = 0
    output = {}
    for qidx, sample in tqdm(metadata.items(), desc="Processing"):
        type_q = sample['type_q']
        if type_q != "sequence":
            continue
        file_path = f"{folder}/step_1_{qidx}.json"
        data = load_json(file_path)
        total_key = len(data['0'].keys())
        
        output[qidx] = {
            'original_question': sample['target_answer_type'],
            'last_subq': data['0'][str(total_key-1)]['target_answer_type']
        }
        
        if output[qidx]['original_question'].lower() != output[qidx]['last_subq'].lower():
            cnt += 1
    print(cnt)
    save_json(output, 'last_subq_sequence_target_ans.json')

if __name__ == "__main__":
    process_sequence_questions()