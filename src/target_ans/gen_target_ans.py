# Import process_batch from pipeline_target_answer_type_batch.py
from pipeline_target_answer_type_batch import process_batch

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

# ===== Load metadata =====
metadata = load_json(r'data\metadata_9000q.json')
def process_sequence_questions(batch_size=10):
    output_res = {}
    sequence_samples = []

    # ===== Lọc câu hỏi loại sequence =====
    for qidx, sample in metadata.items():
        if sample.get("type_q") == "parallel":
            sequence_samples.append((qidx, sample["Question"]))

    print(f"Tổng số câu hỏi sequence: {len(sequence_samples)}")

    # ===== Chạy theo batch =====
    for i in tqdm(range(0, len(sequence_samples), batch_size), desc="Processing"):
        batch_slice = sequence_samples[i:i+batch_size]

        # Tạo dict truyền vào process_batch
        # {qidx: question}
        batch_questions = {qidx: question for qidx, question in batch_slice}

        batch_results = process_batch(batch_questions)
        # batch_results expected format:
        # [{'id': qidx, 'target_answer_type': xxx}, ...]

        for result in batch_results:
            qid = result["id"]
            output_res[qid] = result.get("target_answer_type")
            
        save_json(output_res, "parallel_target_answer_type.json")

    print("Tổng số đã xử lý:", len(output_res))

    # ===== Lưu file =====
    # save_json(output_res, "sequence_target_answer_type.json")
    save_json(output_res, "parallel_target_answer_type.json")
if __name__ == "__main__":
    process_sequence_questions()