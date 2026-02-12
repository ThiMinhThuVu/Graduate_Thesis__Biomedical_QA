import json
import os


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

metadata = load_json(r'output_metadata_mapping.json')

cnt = 0
list_id = []
for qidx, item in metadata.items():
    for cand in item['candidates']: 
        if cand['source'] == 'crawl_ner_wiki' and cand['is_correct'] == False:
            cnt += 1
            list_id.append({qidx: cand})
print(cnt)
save_json(list_id, "crawl_wiki_ner.json")