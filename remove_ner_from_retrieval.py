import os
import json

retrieval_dir = os.path.join("data", "cache", "retrieval")

for filename in os.listdir(retrieval_dir):
    if not filename.endswith(".json"):
        continue
    file_path = os.path.join(retrieval_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    if isinstance(data, dict) and "ner" in data:
        del data["ner"]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    elif isinstance(data, list):
        changed = False
        for item in data:
            if isinstance(item, dict) and "ner" in item:
                del item["ner"]
                changed = True
        if changed:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
