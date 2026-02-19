import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from postprocessing import postprocessing

def load_json(golden_path):
    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)
    return golden_data

dict_golden = load_json(r"data\all_answer_golden_56.json")
meta_mapping = load_json(r"data\mapping_answer_gol.json")
pred_dir = "medhop_results/all_ans"

list_pred = load_json(r'data\test_WH_id_50q.json')

# Setup counters
correct_count = 0
wrong_count = 0
insufficient_count = 0
not_found_file = 0

# Định nghĩa các từ khóa đại diện cho việc không tìm thấy câu trả lời
# Tùy vào hệ thống của bạn, có thể là "not selected", "insufficient", v.v.
INSUFFICIENT_KEYWORDS = ["not selected", "insufficient", "none", "not found"]

for id in list_pred:
    filepath = f"{pred_dir}/R4_step_3b_all_ans_{id}.json"
    
    if os.path.exists(filepath):
        data = load_json(filepath)
        # Lấy kết quả từ final_answer
        final_obj = data.get('final_answer', {})
        
      
        ans = final_obj.get('short_answer', None)

        # 1. TRƯỜNG HỢP INSUFFICIENT / NOT SELECTED
        if ans.lower() == "insufficient information":
            insufficient_count += 1
            continue

        # 2. XỬ LÝ CHUẨN HÓA CÂU TRẢ LỜI
        ans = str(ans).lower().strip()
        
        if ans in meta_mapping:
            ans = meta_mapping[ans]
        else:
            ans = postprocessing(ans)
            ans = meta_mapping.get(ans, ans)
            
        list_golden = dict_golden.get(id, {}).get('True', [])
        
        # 3. KIỂM TRA ĐÚNG/SAI
        if ans in [str(g).lower() for g in list_golden]:
            correct_count += 1
        else:
            wrong_count += 1
            # print(f"ID: {id} - Pred: {ans} - Golden: {list_golden}") # Debug câu sai
    else:
        not_found_file += 1

# --- BÁO CÁO KẾT QUẢ ---
total_processed = correct_count + wrong_count + insufficient_count
print("-" * 30)
print(f"SUMMARY REPORT")
print("-" * 30)
print(f"Total Questions: {len(list_pred)}")
print(f"Correct Answers (TP): {correct_count}")
print(f"Wrong Answers (FP):   {wrong_count}")
print(f"Insufficient (FN/Refused): {insufficient_count}")
if not_found_file > 0:
    print(f"Missing Files: {not_found_file}")
print("-" * 30)
if total_processed > 0:
    accuracy = (correct_count / total_processed) * 100
    print(f"Accuracy (on answered): {accuracy:.2f}%")