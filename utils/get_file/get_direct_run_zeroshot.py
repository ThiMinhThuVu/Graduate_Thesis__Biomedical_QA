import json
import  os

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
data = load_json(r'data\metadata_medhop.json')
o3_1 = load_json(r'data\cache\O3_zeroshot_ans_cache.json')
o3_2 = load_json(r'data\cache\O3_zeroshot_ans_cache_direct.json')

dict_output = {}
for qidx, item in data.items():
    if item['type_q'] != 'direct':
        continue
    if qidx in o3_1 or qidx in o3_2:
        continue
    dict_output[qidx] = item
    
save_json(dict_output, r'zero-shot-o3\direct\list_id_miss_o3_zeroshot.json')
print(len(dict_output.keys()))