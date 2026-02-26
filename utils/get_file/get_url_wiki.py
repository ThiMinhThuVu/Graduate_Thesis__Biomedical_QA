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


meta = load_json(r'2wiki\cache\dict_searchkw\compositional_dict_search_kw.json')
input_data2 = load_json(r'2wiki\cache\dict_searchkw\comparison_dict_search_kw.json')
input_data3 = load_json(r'2wiki\cache\dict_searchkw\inference_dict_search_kw.json')
input_data4 = load_json(r'2wiki\cache\dict_searchkw\bridge_comparison_dict_search_kw.json')

meta.update(input_data2)
meta.update(input_data3)
meta.update(input_data4)   
    
cache_searchkw = load_json(r'2wiki\cache\hash_id.json')
cache_url = load_json(r'2wiki\cache\mapping\url2hash.json')
print(len(cache_url.keys()))
data = load_json(r'data\cache\mapping\entity2wiki.json')
list_ner = load_json(r'data/ner/SEQ/spacy_entity_9000q_search.json')

# all_ner = set()
# for ner in list_ner:
#     if ner.lower() not in data:
#         all_ner.add(ner.lower())

# save_json(list(all_ner), 'data/ner/SEQ/spacy_entity_9000q_search_filter.json' )
# exit()
# cache_url = {}
# for url, hashid in cache_url.items():
#     url = url.lower()
#     if url not in cache_url:
#         cache_url[url] = hashid
    
# save_json(cache_url, r'2wiki\cache\mapping\url2hash.json')


cache_old =load_json(r'2wiki/crawl/crawl_wiki_2wiki_1.json')

BASE = '2wiki/cache/IR_20'
run = 2
list_url_need = {}


for qidx, list_search_kw in tqdm(meta.items(), desc="get searchkw"):
    for search_kw in list_search_kw:
        search_kw = search_kw.lower().strip()
        if search_kw in cache_searchkw:
            hashid = cache_searchkw[search_kw]
            search_file = f"{BASE}/{hashid}"
            if os.path.exists(search_file):
                wiki_search = load_json(search_file)
                if 'wiki' not in wiki_search:
                    print(search_file)
                    continue
                wiki_search = wiki_search['wiki']['search_results']
                cnt = 0
                for item in wiki_search:
                    if cnt == 3:
                        break
                    link_raw = item['link']
                    link = link_raw.lower().strip()
                    if 'list of' in link or 'lists of' in link or 'wikiroject' in link or 'wikipedia talk' in link:
                        continue
                    cnt += 1
                    if link in cache_url or link_raw in cache_old:
                        continue
                    else:
                        hash_url = make_cache_filename(link)
                        cache_url[link] = hash_url
                        list_url_need[link_raw] =hash_url
                        
                        
print(len(list_url_need.keys()))
output_file = f'2wiki/crawl/crawl_wiki_2wiki_{run}.json'
save_json(list_url_need,output_file )
print(output_file)
                    
                
cache_url.update(cache_old)
save_json(cache_url, r"2wiki\cache\mapping\url2hash.json")   
print(len(cache_url.keys()))
