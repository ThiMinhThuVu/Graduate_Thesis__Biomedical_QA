import os
import re
import json

run = 'R11'
q_type = 'inference'

wiki_result_directory = f'wiki_{q_type}_results/all_ans' #r"E:\Test_2wiki\test_wiki_result_v0\all_ans" # f'wiki_{q_type}_results/all_ans'
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
    # predict = data[last_step_key]['short_answer']
    predict = ''
    retrieval = data[last_step_key]['retrieval_information']
    if 'normalized_v2' in data[last_step_key] and data[last_step_key]['normalized_v2'] is not None:
        verify = data[last_step_key]['normalized_v2']['verify']
        if len(verify) > 0 and 'parent_label' in verify[0]:
            predict = verify[0]['parent_label']
            
        if predict == '':
            list_predict = []
            dict_parent = data[last_step_key]['normalized_v2']['parent']
            for parent_id, parent in dict_parent.items():
                if len(parent) > 0 and 'propertyLabel' in parent[0]:
                    list_predict.append(parent[0]['propertyLabel']['value'])
                    
            # if len(list_predict) > 0:
            #     predict = list_predict[0]
            for p in list_predict:
                if p in retrieval:
                    predict = p
                    break        

    if predict == '' and 'normalized_ans' in data[last_step_key]:
        predict = data[last_step_key]['normalized_ans']['full_name']
        if predict != '':
            if predict not in retrieval:
                predict = ''
                
        if predict == '':
            predict = data[last_step_key]['normalized_ans']['suffix']
            # if predict != '':
            #     if predict not in retrieval:
            #         predict = ''
        
        if predict == '':
            predict = data[last_step_key]['normalized_ans']['name']
            # if predict != '':
            #     if predict not in retrieval:
            #         predict = ''
            
        if predict == '':
            predict = data[last_step_key]['normalized_ans']['prefix']
            # if predict != '':
            #     if predict not in retrieval:
            #         predict = ''
    
   
    if predict == '' or predict not in retrieval:
        predict = data[last_step_key]['short_answer']
        
    predict_dict['answer'][id] = predict

with open(pred_path, 'w', encoding='utf-8') as f:
    json.dump(predict_dict, f) 