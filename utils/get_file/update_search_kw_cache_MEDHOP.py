import os
import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
# Path to the folder containing JSON files
json_folder = os.path.join('data', 'cache', 'IR_20')
cache_file = os.path.join('data', 'cache', 'mapping', 'hash_id_IR_20.json')

# Load existing cache if it exists
# if os.path.exists(cache_file):
#     with open(cache_file, 'r', encoding='utf-8') as f:
#         search_kw_cache = json.load(f)
# else:
search_kw_cache = load_json(cache_file)

# output = {}
# for url, ID in search_kw_cache.items():
#     url = url.lower()
#     output[url] = ID
# save_json(output, cache_file)
# print(len(output.keys()))
# exit()


# Iterate through all JSON files in the folder
for filename in os.listdir(json_folder):
    if filename.endswith('.json'):
        hashID = filename # Remove .json extension
        file_path = os.path.join(json_folder, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Check if hashID is already in cache
            if hashID not in search_kw_cache.values():
                # Try to get search_kw from google or wiki
                search_kw = None
                search_kw = data['google']['search_kw'].lower().strip()
               
                if search_kw not in search_kw_cache:
                    search_kw_cache[search_kw] = hashID
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Save updated cache
with open(cache_file, 'w', encoding='utf-8') as f:
    json.dump(search_kw_cache, f, ensure_ascii=False, indent=2)

print(f"Cache updated. Total entries: {len(search_kw_cache)}")
