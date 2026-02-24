import json
import  os

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
BASE = "medhop_preprocessing/step1"
meta = load_json('data/metadata_medhop_9000q_classified.json')

cnt = 0
dict_output = {}
for i, (qidx, item) in enumerate(meta.items()):    
    if item['predicted_class'] in ['direct', 'sequence']:
        file_data = f"{BASE}/1/step_1_{qidx}.json"
        if os.path.exists(file_data):
            data = load_json(file_data)
            search_keyword = data['0']['0']['search_keyword']            
            dict_output[qidx] = [search_keyword]
            cnt += 1
    else:
        dict_output[qidx] = []
        for qi in ['1',  '2', '3', '4']:
            file_data = f"{BASE}/{qi}/step_1_{qidx}.json"
            if os.path.exists(file_data):
                data = load_json(file_data)
                search_keyword = data['0']['0']['search_keyword']  
                dict_output[qidx].append(search_keyword)
                cnt += 1

save_json(dict_output, 'data/search_kw_9000q_medhop.json')
print("Total search of 9000q: ", cnt)
        

            

            
    
    