import json 
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from target_ans.pipeline_target_answer_type_batch import process_batch


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
# metadata = load_json(r'data\metadata_medhop.json')
metadata = load_json(r'data\metadata_medhop_9000q_classified.json')

# Import process_batch from pipeline_target_answer_type_batch.py

# def process_sequence_questions():
#     cnt = 0
#     for qidx, sample in tqdm(metadata.items(), desc="Processing"):
#         type_q = sample['type_q']
#         if type_q != "sequence": # "sequence":
#             continue
#         file_path = f"{folder}/step_1_{qidx}.json"
        
#         if os.path.exists(file_path):
#             data = load_json(file_path)
#             batch_questions = {step: item['subquestion'] for step, item in data['0'].items()}
#             batch_results = process_batch(batch_questions)
#             # batch_results is a list of dicts with keys: id, target_answer_type
#             for result in batch_results:
#                 step = result['id']
#                 target_ans = result['target_answer_type']
#                 data['0'][str(step)]['target_answer_type'] = target_ans
#             save_json(data, file_path)
#             print("file: ", file_path)
#             cnt += 1
#     print(cnt)

def process_sequence_questions_batching(limit_range=[4500, 9000]):
    batch_size = 10
    current_batch_qidxs = [] 
    global_batch_input = {} 
    
    cnt = 0
    # 1. Lấy tất cả các item thỏa mãn điều kiện trước
    all_sequence_items = [(k, v) for k, v in metadata.items() if v.get('predicted_class') == "direct"]
    
    start, end = limit_range
    sequence_items = all_sequence_items[start:end]
    
    print(f"Bắt đầu xử lý từ index {start} đến {end}. Tổng cộng: {len(sequence_items)} items.")
    for i, (qidx, sample) in enumerate(tqdm(sequence_items, desc="Batch Processing")):
        file_path = f"{folder}/step_1_{qidx}.json"
        
        if os.path.exists(file_path):
            data = load_json(file_path)
            if data['0']['0']['target_answer_type'] != None:
                continue
            
            # Gom toàn bộ step của qidx này vào global batch
            for step, item in data['0'].items():
                # Tạo key duy nhất: "qidx|step" để sau này dễ tách
                unique_key = f"{qidx}|{step}"
                global_batch_input[unique_key] = item['subquestion']
            
            current_batch_qidxs.append((qidx, data, file_path))

        # Khi đủ 3 qidx HOẶC là câu cuối cùng thì bắt đầu xử lý
        if len(current_batch_qidxs) == batch_size or i == len(sequence_items) - 1:
            if not global_batch_input:
                continue

            # 1. Gọi batch processing cho tất cả subquestions gom được
            batch_results = process_batch(global_batch_input)

            # 2. Map kết quả trả về lại cho từng data object
            # Giả sử batch_results trả về list các dict: {'id': 'qidx|step', 'target_answer_type': ...}
            for result in batch_results:
                u_id = result['id']
                target_ans = result['target_answer_type']
                
                # Tách qidx và step từ unique_key
                res_qidx, res_step = u_id.split('|')
                
                # Tìm đúng data object trong batch hiện tại để update
                for b_qidx, b_data, b_path in current_batch_qidxs:
                    if b_qidx == res_qidx:
                        b_data['0'][res_step]['target_answer_type'] = target_ans

            # 3. Lưu tất cả file trong batch này
            for b_qidx, b_data, b_path in current_batch_qidxs:
                save_json(b_data, b_path)
                cnt += 1
                print(b_path)

            # 4. Reset buffer cho batch tiếp theo
            current_batch_qidxs = []
            global_batch_input = {}

    print(f"Hoàn thành! Tổng số file đã xử lý: {cnt}")

if __name__ == "__main__":
    # process_sequence_questions()
    process_sequence_questions_batching()