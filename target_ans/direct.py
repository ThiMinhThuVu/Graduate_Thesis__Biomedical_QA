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
        
folder = "medhop_preprocessing/step1/4"
metadata = load_json(r'data\metadata_medhop.json')

# Import process_batch from pipeline_target_answer_type_batch.py
from pipeline_target_answer_type_batch import process_batch

def process_sequence_questions():
    cnt = 0
    for qidx, sample in tqdm(metadata.items(), desc="Processing"):
        type_q = sample['type_q']
        if type_q != "parallel": # "sequence":
            continue
        file_path = f"{folder}/step_1_{qidx}.json"
        
        if os.path.exists(file_path):
            data = load_json(file_path)
            batch_questions = {step: item['subquestion'] for step, item in data['0'].items()}
            batch_results = process_batch(batch_questions)
            # batch_results is a list of dicts with keys: id, target_answer_type
            for result in batch_results:
                step = result['id']
                target_ans = result['target_answer_type']
                data['0'][str(step)]['target_answer_type'] = target_ans
            save_json(data, file_path)
            print("file: ", file_path)
            cnt += 1
    print(cnt)

if __name__ == "__main__":
    process_sequence_questions()