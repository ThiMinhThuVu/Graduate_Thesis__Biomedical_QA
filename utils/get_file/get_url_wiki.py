import json
import  os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm import tqdm
from utils.hash import make_cache_filename
from utils.crawl_wikipage import crawl_wikipedia_page

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# output = {}
# metadata = load_json(r'data\cache\mapping\url2hash.json')
# for url, hashid in metadata.items():
#     url = url.lower()
#     output[url] = hashid
        
# save_json(output, r'data\cache\mapping\url2hash.json')

data = load_json(r'data\cache\mapping\entity2wiki.json')
for i in range(9):
    filepath = fr'data\cache\mapping\entity\entity2wiki{i}.json'
    data.update(load_json(filepath))
    
save_json(data, r'data\cache\mapping\entity2wiki_2.json')
exit()
     
     
metadata = load_json(r'data\ner\entity_extraction_merge_9000.json')
output = {}
for qidx, item in metadata.items():
    output[qidx] = {
        'QIDX': qidx,
        'Question': item['original_question'],
        'type_q': item['type_q'],
        'entities_q': item['entities_q']
    }
    
save_json(output, r'data/metadata_9000q.json')
exit()
# cache_url = load_json(r"data\cache\mapping\url2hash.json")

# for i in range(6,1):
#     # file_path = f"data/CRAWL/ner/url2hash{i}.json" 
#     file_path = f"data/ner/crawl_wiki_9000q_{i}.json"
#     if os.path.exists(file_path):   
#         print(len(cache_url.keys()))

#         cache_url.update(load_json(file_path))
        
# print(len(cache_url.keys()))
# save_json(cache_url, r"data\cache\mapping\url2hash.json")   
         
# cache_url = load_json(r"data\cache\mapping\entity2wiki.json")

# for i in range(8):
#     file_path = f"data/CRAWL/ner/entity2wiki{i}.json"  
#     if os.path.exists(file_path):   
#         print(len(cache_url.keys()))

#         cache_url.update(load_json(file_path))
        
# print(len(cache_url.keys()))
# save_json(cache_url, r"data\cache\mapping\entity2wiki.json")   
# exit()

meta = load_json(r'data\search_kw_9000q_medhop.json')
cache_searchkw = load_json('data/cache/mapping/hash_id.json')
cache_url = load_json(r'data\cache\mapping\url2hash.json')

cache_old = load_json(r'data/ner/crawl_wiki_9000q_1.json')
cache_old.update(load_json(r'data/ner/crawl_wiki_9000q_2.json'))
cache_old.update(load_json(r'data/ner/crawl_wiki_9000q_3.json'))
cache_old.update(load_json(r'data/ner/crawl_wiki_9000q_4.json'))
cache_old.update(load_json(r'data/ner/crawl_wiki_9000q_5.json'))
cache_old.update(load_json(r'data/ner/crawl_wiki_9000q_6.json'))

BASE = 'data/cache/IR_20'
run = 6
list_url_need = {}

print(len(cache_old.keys()))

# for qidx, list_search_kw in tqdm(meta.items(), desc="get searchkw"):
#     for search_kw in list_search_kw:
#         search_kw = search_kw.lower().strip()
#         if search_kw in cache_searchkw:
#             hashid = cache_searchkw[search_kw]
#             search_file = f"{BASE}/{hashid}"
#             if os.path.exists(search_file):
#                 wiki_search = load_json(search_file)
#                 if 'wiki' not in wiki_search:
#                     print(search_file)
#                     continue
#                 wiki_search = wiki_search['wiki']['search_results']
#                 cnt = 0
#                 for item in wiki_search:
#                     if cnt == 3:
#                         break
#                     link_raw = item['link']
#                     link = link_raw.lower().strip()
#                     if 'list of' in link or 'lists of' in link or 'wikiroject' in link or 'wikipedia talk' in link:
#                         continue
#                     cnt += 1
#                     if link in cache_url or link_raw in cache_old:
#                         continue
#                     else:
#                         hash_url = make_cache_filename(link)
#                         cache_url[link] = hash_url
#                         list_url_need[link_raw] =hash_url
                        
#                         # crawled_data = crawl_wikipedia_page(link)
#                         # crawled_data_output = {
#                         #     'crawl_wiki': {
#                         #         'html_page': {
#                         #             link: crawled_data
#                         #         }
#                         #     }
#                         # }
#                         # file_crawl = f"data/cache/wiki_pages_tool_1/{hash_url}"
#                         # save_json(crawled_data_output, file_crawl)
#                         # print(file_crawl)
                        
# print(len(list_url_need.keys()))
# save_json(list_url_need, f'data/ner/crawl_wiki_9000q_{run}.json')
                    
                
cache_url.update(cache_old)
save_json(cache_url, r"data\cache\mapping\entity2wiki.json")   
print(len(cache_url.keys()))
