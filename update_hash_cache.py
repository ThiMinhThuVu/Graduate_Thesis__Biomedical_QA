import os
import json


def normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())

retrieval_folder = os.path.join('data', 'cache', 'IR_20')
hash_cache_path = os.path.join('data', 'cache', 'hash_id.json')

def update_hashid(hash_cache_path):
    retrieval_folder = os.path.join('data', 'cache', 'IR_20')
    hash_cache_path = os.path.join('data', 'cache', 'hash_id.json')

    # Load cache if exists
    if os.path.exists(hash_cache_path):
        with open(hash_cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}

    # Scan retrieval folder
    for filename in os.listdir(retrieval_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(retrieval_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
            query = data.get('google', {}).get('search_kw', None)
            if query is not None:
                query = normalize_query(query)
                if query not in cache:
                    cache[query] = filename

    # Save updated cache
    with open(hash_cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"Updated {hash_cache_path} with new queries.")


def update_hashid_crawl_wiki():
    retrieval_folder = os.path.join('data', 'cache', 'crawl_wikipedia')
    hash_cache_path = os.path.join('data', 'cache', 'hash_id_crawl_wiki.json')

    # Load cache if exists
    if os.path.exists(hash_cache_path):
        with open(hash_cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}

    # Scan retrieval folder
    for filename in os.listdir(retrieval_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(retrieval_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
            
            query = next(iter(data.get('crawl_wiki', {}).get('html_page', {})), None)
            if query is not None:
                query = normalize_query(query)
                if query not in cache:
                    cache[query] = filename

    # Save updated cache
    with open(hash_cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"Updated {hash_cache_path} with new queries.")
    
update_hashid(hash_cache_path)