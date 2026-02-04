import os
import json

# Define folder paths
folder1 = os.path.join('data', 'cache', 'retrieval')
folder2 = os.path.join('data', 'cache', 'filter_wikipedia')

# Ensure folder2 exists
os.makedirs(folder2, exist_ok=True)

# Iterate over all JSON files in folder1
for filename in os.listdir(folder1):
    if filename.endswith('.json'):
        file_path = os.path.join(folder1, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        # Check and process 'crawl_wiki'
        if 'crawl_wiki' in data:
            # Save crawl_wiki to folder2, wrapped in {"crawl_wiki": ...}
            crawl_wiki_path = os.path.join(folder2, filename)
            try:
                with open(crawl_wiki_path, 'w', encoding='utf-8') as f:
                    json.dump({'crawl_wiki': data['crawl_wiki']}, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error saving crawl_wiki for {filename}: {e}")
                continue
            # Remove 'crawl_wiki' from original data
            del data['crawl_wiki']
            # Save the updated file back to folder1
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error updating {filename}: {e}")
