import json
import  os

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
data = load_json(r'2wiki\meta_demo_3000q.json')

output = {}
cnt = 0
for qidx, item in data.items():
    if item["type"] != 'comparison':
        continue
    if cnt == 200:
        break
    cnt += 1
    output[qidx] = item

save_json(output, r'2wiki\meta_demo_comparison_200q.json')