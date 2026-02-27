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

# if __name__ == "__main__":
#     cache_IR20 = "E:/2wiki/2wiki/cache/IR_20"
#     # file_path_cache = r"2wiki\cache\hash_id.json"
#     # cache_hashid = load_json(file_path_cache)
#     # input_data = load_json(r'2wiki\cache\dict_searchkw\compositional_dict_search_kw.json')
#     # input_data = load_json(r"E:\2wiki\2wiki\cache\dict_searchkw\comparison_dict_search_kw.json")
#     input_data = load_json(r'E:\2wiki\2wiki\cache\dict_searchkw\bridge_comparison_dict_search_kw.json')
#     # input_data4 = load_json(r'2wiki\cache\dict_searchkw\bridge_comparison_dict_search_kw.json')
    
    
#     from tqdm import tqdm
    
#     for i, (qidx, list_searchkw) in enumerate(tqdm(input_data.items(), desc="Processing")):
#         for search_kw in list_searchkw:
#             # hashid, cache_hashid = register_query_cache(search_kw, cache_hashid)
#             hashid = make_cache_filename(search_kw.lower())
#             filepath = f"{cache_IR20}/{hashid}"
#             if os.path.exists(filepath):
#                 data = load_json(filepath)
#                 if 'google' not in data:
#                     output_gg = retrieval_google(search_kw, filepath)

#                 continue
                
#             output_gg = retrieval_google(search_kw, filepath)
#             if output_gg is None:
#                 print(filepath)
#                 exit()

import shutil
from tqdm import tqdm
if __name__ == "__main__":
    cache_IR20 = "E:/2wiki/2wiki/cache/IR_20"
    input_path = r'E:\2wiki\2wiki\cache\dict_searchkw\bridge_comparison_dict_search_kw.json'
    
    # 1. Tự động xác định tên thư mục đích dựa trên tên file input
    # Lấy "bridge_comparison_dict_search_kw" từ đường dẫn
    type_name = os.path.basename(input_path).replace(".json", "")
    dest_folder = os.path.join(os.path.dirname(cache_IR20), "extracted_json", type_name)
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(dest_folder, exist_ok=True)
    print(f"--- Thư mục đích: {dest_folder} ---")

    input_data = load_json(input_path)
    
    copy_count = 0
    for i, (qidx, list_searchkw) in enumerate(tqdm(input_data.items(), desc="Processing & Copying")):
        for search_kw in list_searchkw:
            hashid = make_cache_filename(search_kw.lower())
            source_filepath = os.path.join(cache_IR20, hashid)
            dest_filepath = os.path.join(dest_folder, hashid)

            # Kiểm tra file gốc có tồn tại không
            if os.path.exists(source_filepath):
                # Logic cũ của bạn: Kiểm tra nội dung json
                data = load_json(source_filepath)
                if 'google' not in data:
                    retrieval_google(search_kw, source_filepath)
                
                # --- LOGIC COPY FILE ---
                # Chỉ copy nếu file chưa tồn tại ở đích (để tiết kiệm thời gian)
                if not os.path.exists(dest_filepath):
                    shutil.copy2(source_filepath, dest_filepath) # copy2 giữ nguyên metadata
                    copy_count += 1
            else:
                # Nếu không tồn tại thì có thể chạy retrieval để tạo mới file đó
                output_gg = retrieval_google(search_kw, source_filepath)
                if output_gg is not None:
                    # Sau khi tạo mới xong thì copy sang luôn
                    shutil.copy2(source_filepath, dest_filepath)
                    copy_count += 1
                else:
                    print(f"Lỗi: Không tìm thấy và không thể crawl: {source_filepath}")
                    # exit() # Tùy bạn có muốn dừng chương trình không

    print(f"--- Hoàn thành! Đã copy {copy_count} file vào {dest_folder} ---")