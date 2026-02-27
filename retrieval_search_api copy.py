from src.gpt_request import search_google
import json
import re
import os
import math
from utils.hash import make_cache_filename

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
def retrieval_google(search_kw, filepath, max_results=20):
    if os.path.exists(filepath):
        output = load_json(filepath)
        if 'google' in output:
            result = output['google']['search_results']
            if result == []:
                print("[API SEARCH] HẾT LƯỢT SEARCHs")
                print("Search kw: ", search_kw)
                return None
            
            print(f"Cache hit Google Search. Loading from file.")
            return result
    else:
        output = {}
            
    max_retries = 3
    for attempt in range(max_retries):
        result = search_google(search_kw, max_results=max_results)
        if result != []:
            break
        else:
            print(f"[API SEARCH] HẾT LƯỢT SEARCHs (attempt {attempt+1}/{max_retries})")
            print("Search kw: ", search_kw)
            
    if result == []:
        return None
          
    output['google'] = {
        'search_kw': search_kw,
        "search_results": result                
    }
  
    save_json(output, filepath)        
    return result

def retrieval_wiki(search_kw, filepath, max_results=10):
    if os.path.exists(filepath):
        output = load_json(filepath)
        if 'wiki' in output:
            print(f"Cache hit Wikipedia Search. Loading from file.")
            return output['wiki']['search_results']
    else:
        output = {}
     
    result = search_google(search_kw + " site:wikipedia.org", max_results=max_results)           
    
    output['wiki'] = {
        'search_kw': search_kw,
        "search_results": result                
    }
  
    save_json(output, filepath)        
    return result

if __name__ == "__main__":
    cache_IR20 = "E:/2wiki/2wiki/cache/IR_20"
    # file_path_cache = r"2wiki\cache\hash_id.json"
    # cache_hashid = load_json(file_path_cache)
    # input_data = load_json(r'2wiki\cache\dict_searchkw\compositional_dict_search_kw.json')
    # input_data = load_json(r"E:\2wiki\2wiki\cache\dict_searchkw\comparison_dict_search_kw.json")
    input_data = load_json(r'E:\2wiki\2wiki\cache\dict_searchkw\bridge_comparison_dict_search_kw.json')
    # input_data4 = load_json(r'2wiki\cache\dict_searchkw\bridge_comparison_dict_search_kw.json')
    
    
    from tqdm import tqdm
    
    for i, (qidx, list_searchkw) in enumerate(tqdm(input_data.items(), desc="Processing")):
        for search_kw in list_searchkw:
            # hashid, cache_hashid = register_query_cache(search_kw, cache_hashid)
            hashid = make_cache_filename(search_kw.lower())
            filepath = f"{cache_IR20}/{hashid}"
            if os.path.exists(filepath):
                continue
            output_gg = retrieval_google(search_kw, filepath)
            if output_gg is None:
                print(filepath)
                exit()
                