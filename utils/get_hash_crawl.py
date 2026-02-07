import os
import json
from utils.hash import make_cache_filename, normalize_query
from utils.util import save_json, load_pickle, load_json, wiki_pre_processing
from utils.crawl_wikipage import crawl_wikipedia_page, crawl_wikipedia_tool_2

def crawl_wikipedia():
    inputdata = load_json(r'data\cache\IR_20_search_results.json')
    dst_folder = "e:/MAS_UETQuintet/data/cache/crawl_wikipedia"
    dst_folder_2 = "e:/MAS_UETQuintet/data/cache/crawl_wikipedia_tool_2"
    index_file = r'data\cache\hash_id_crawl_wiki.json'
    hash_index = load_json(index_file)
    
    for idx, (link, item) in enumerate(inputdata.items()):
        # if idx > 1000:
        #     continue
        original_link = link
        title = item['title']
        language = item['language']

        norm_link = normalize_query(original_link.lower())
        if norm_link in hash_index:
            hash_file_name = hash_index[norm_link]
        else:
            hash_file_name = make_cache_filename(norm_link)
            hash_index[norm_link] = hash_file_name
            save_json(hash_index, index_file)

        out_path = os.path.join(dst_folder, hash_file_name)
        if os.path.exists(out_path):
            continue
        else:
            wiki_content = {}
            if 'list_' in norm_link or 'lists_' in norm_link:
                continue
            else:
                print(f"Crawling Wikipedia page: {original_link}")
                wiki_content = crawl_wikipedia_page(original_link)
                # Nếu lỗi hoặc content rỗng thì gọi crawl_wikipedia_tool_2
                if (
                    wiki_content is None
                    or 'content' not in wiki_content
                    or not wiki_content['content']
                ):
                    print(f"Fallback to crawl_wikipedia_tool_2 for: {original_link}")
                    wiki_content = crawl_wikipedia_tool_2(title, language, original_link)
                    out_path = os.path.join(dst_folder_2, hash_file_name) #gọi tool2 lưu folder 2
                    
            
            out = {
                "crawl_wiki": {
                    "html_page": {
                        norm_link: wiki_content
                    }
                }
            }

            with open(out_path, "w", encoding="utf-8") as fout:
                json.dump(out, fout, ensure_ascii=False, indent=2)
            
            print("file: ", out_path)

crawl_wikipedia()