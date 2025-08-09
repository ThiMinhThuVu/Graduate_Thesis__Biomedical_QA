import os
import re
import json

wiki_result_directory = f'run/evidence/wiki_results/all_ans'
pred_path = f'run/evidence/pred.json'

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
    predict_dict['answer'][id] = data['final_answer']['short_answer']
    predict_dict['evidence'][id] = data['evidence']

for id, evidences in predict_dict['evidence'].items():
    predict_dict['evidence'][id] = [[ev if ev is not None else '' for ev in evidence] for evidence in evidences]

with open(pred_path, 'w', encoding='utf-8') as f:
    json.dump(predict_dict, f) 