import os
import re
import json

run = 'R11'
q_type = 'inference'

wiki_result_directory = f'wiki_{q_type}_v4_results/all_ans'
pred_path = f'evaluate/run/{run}/pred_{q_type}.json'
os.makedirs(os.path.dirname(pred_path), exist_ok=True)

def load_json(fname, mode="r", encoding="utf-8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)

def filename_format(filename):
    ans_file_pattern = r'R11_step_3b_ans_(.+)\.json'
    all_ans_file_pattern = r'R11_step_3b_all_ans_(.+)\.json'

    ans_match = re.match(ans_file_pattern, filename)
    all_ans_file_match = re.match(all_ans_file_pattern, filename)

    if ans_match:
        return 'ans'
    elif all_ans_file_match:
        return 'all_ans'
    return None

def extract_id(filename):
    pattern1 = r'R11_step_3b_ans_(.+)\.json'
    pattern2 = r'R11_step_3b_all_ans_(.+)\.json'
    
    match1 = re.match(pattern1, filename)
    match2 = re.match(pattern2, filename)
    
    if match1:
        return match1.group(1).strip()
    elif match2:
        return match2.group(1).strip()
    return None

predict_dict = {
    'answer': {},
    'sp': {},
    'evidence': {}
}

for filename in os.listdir(wiki_result_directory):
    id = extract_id(filename)
    file_path = os.path.join(wiki_result_directory, filename)
    
    data = load_json(file_path)
    if filename_format(filename) == 'ans':
        last_step_key = max(data.keys(), key=int)
    elif filename_format(filename) == 'all_ans':
        last_step_key = 'final_answer'
    else:
        print(f'Có lỗi nhận dạng file: {filename}')
    predict = data[last_step_key]['short_answer']
    predict_dict['answer'][id] = predict

with open(pred_path, 'w', encoding='utf-8') as f:
    json.dump(predict_dict, f) 