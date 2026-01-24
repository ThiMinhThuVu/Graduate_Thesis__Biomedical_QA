import os
import json

def build_hash_id_from_retrieval_folder(folder_path):
    hash_id = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Ưu tiên lấy search_kw từ google, nếu không có thì thử wiki
                search_kw = None
                if 'google' in data and 'search_kw' in data['google']:
                    search_kw = data['google']['search_kw']
                elif 'wiki' in data and 'search_kw' in data['wiki']:
                    search_kw = data['wiki']['search_kw']
                if search_kw:
                    # Normalize: remove extra spaces
                    norm_search_kw = ' '.join(search_kw.strip().split())
                    norm_search_kw = norm_search_kw.lower()
                    hash_id[norm_search_kw] = filename #.replace('.json', '')
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return hash_id

# Example usage:
folder = 'data/cache/retrieval'
hash_id = build_hash_id_from_retrieval_folder(folder)
with open('data/cache/hash_id.json', 'w', encoding='utf-8') as f:
    json.dump(hash_id, f, ensure_ascii=False, indent=2)
