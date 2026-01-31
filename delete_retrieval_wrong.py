import os
import json

# Đường dẫn tới thư mục chứa các file json
folder = r"e:/MAS_UETQuintet/data/cache/retrieval/"

count = 0
for filename in os.listdir(folder):
    if filename.endswith('.json'):
        filepath = os.path.join(folder, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'google' in data and isinstance(data['google'], dict) and data['google'].get('search_results', None) == []:
                os.remove(filepath)
                count += 1
                print(f"Deleted: {filename}")
        except Exception as e:
            print(f"Error with {filename}: {e}")
print(f"Total deleted: {count}")
