import os
import json
from utils.hash import make_cache_filename, normalize_query
from utils.util import save_json, load_pickle, load_json, wiki_pre_processing
from utils.crawl_wikipage import crawl_wikipedia_page, crawl_wikipedia_tool_2

def crawl_wikipedia():
    inputdata = load_json(r'data\cache\IR_20_search_results_YN_direct.json')
    dst_folder = "e:/MAS_UETQuintet/data/cache/crawl_wikipedia"
    dst_folder_2 = "e:/MAS_UETQuintet/data/cache/crawl_wikipedia_tool_2"
    index_file = r'data\cache\hash_id_crawl_wiki.json'
    hash_index = load_json(index_file)
    
    for idx, (link, item) in enumerate(inputdata.items()):
        print(idx)
        if idx < 2240:
            continue
        original_link = link
        title = item['title']
        language = item['language']

        norm_link = normalize_query(original_link.lower())
        if 'list_' in norm_link or 'lists_' in norm_link:
            continue
            
        if norm_link in hash_index:
            hash_file_name = hash_index[norm_link]
        else:
            hash_file_name = make_cache_filename(norm_link)
            hash_index[norm_link] = hash_file_name
            save_json(hash_index, index_file)

        out_path_1 = os.path.join(dst_folder, hash_file_name)
        out_path_2 = os.path.join(dst_folder_2, hash_file_name) #gọi tool2 lưu folder 2
        if os.path.exists(out_path_1):
            print("file1: ", out_path_1)
            pass
        else:
            wiki_content = {}
            # print(f"Crawling Wikipedia tool 1: {original_link}")
            wiki_content = crawl_wikipedia_page(original_link)
            out_1 = {
                "crawl_wiki": {
                    "html_page": {
                        norm_link: wiki_content
                    }
                }
            }

            with open(out_path_1, "w", encoding="utf-8") as fout:
                json.dump(out_1, fout, ensure_ascii=False, indent=2)
            
            print("file 1: ", out_path_1)
           
        if os.path.exists(out_path_2):
            pass
        else:
            print(f"crawl_wikipedia_tool_2 for: {original_link}")
            wiki_content = crawl_wikipedia_tool_2(title, language, original_link)
              
            out_2 = {
                "crawl_wiki": {
                    "html_page": {
                        norm_link: wiki_content
                    }
                }
            }

            with open(out_path_2, "w", encoding="utf-8") as fout:
                json.dump(out_2, fout, ensure_ascii=False, indent=2)
            
            print("file 2: ", out_path_2)

crawl_wikipedia()