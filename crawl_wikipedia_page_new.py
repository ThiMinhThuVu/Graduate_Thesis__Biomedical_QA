import json
import os
from utils.crawl_wikipage import crawl_wikipedia_page, crawl_wikipedia_tool_2

BASE_1 = "data/cache/crawl_wikipedia"
BASE_DIR = "data/cache/crawl_wikipedia_tool_1_to_2"
cache_hashid_crawl_path = "data\cache\hash_id_crawl_wiki.json"

CONFIG = "TOOL_1" # "TOOL_2"

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

  
import re
import json
from urllib.parse import urlparse
# import urllib.parse

def normalize_title_multilang(title):
    # Loại bỏ các hậu tố Wikipedia ở nhiều ngôn ngữ
    patterns = [
        r" - wikipedia.*$",  # tiếng Anh
        r"- 维基百科.*$",     # tiếng Trung
        r"- Википедия.*$",   # tiếng Nga
        r"- Wikipédia.*$",   # tiếng Pháp/Bồ Đào Nha
        r"- Вікіпедія.*$",   # tiếng Ukraina
        r"- Вікіпедыя.*$",   # tiếng Belarus
        r"- ויקיפדיה.*$",    # tiếng Hebrew
        r"- ויקיפדיה.*$",    # tiếng Hebrew
        r"- ويكيبيديا.*$",    # tiếng Ả Rập
        r"- ויקיפדיה.*$",    # tiếng Hebrew
        r"- Wikipedia.*$",   # fallback
    ]
    for pat in patterns:
        title = re.sub(pat, "", title, flags=re.IGNORECASE).strip()
    return title


def normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())

cache_crawl_hash = load_json(cache_hashid_crawl_path)

cnt = 0
for link, hashid_name in cache_crawl_hash.items():
    wiki_content = {}
    f_1 = f"{BASE_1}/{hashid_name}"
    out_path = os.path.join(BASE_DIR, hashid_name)
    if os.path.exists(out_path):
        continue
    
    if os.path.exists(f_1):
        data_crawl = load_json(f_1)
        if data_crawl is not None:
            html = data_crawl.get('crawl_wiki', {}).get('html_page', {}).get(link, None)
            if html is not None:
                title = normalize_title_multilang(html['title'])
                netloc = urlparse(link).netloc
                page_language = netloc.split('.')[0] if netloc.endswith('wikipedia.org') else ''
                if CONFIG == "TOOL_1":
                    wiki_content = crawl_wikipedia_page(title, page_language, link, base_delay=5)
                elif CONFIG == "TOOL_2":
                    wiki_content = crawl_wikipedia_tool_2(title, page_language, link, base_delay=5)
                
                cnt += 1
                
        out = {
            "crawl_wiki": {
                "html_page": {
                    link: wiki_content
                }
            }
        }
        with open(out_path, "w", encoding="utf-8") as fout:
            json.dump(out, fout, ensure_ascii=False, indent=2)
    
    print("file: ", out_path)
    print("success: ", cnt)