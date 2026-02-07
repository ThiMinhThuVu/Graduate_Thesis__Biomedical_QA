import json
import os
from utils.crawl_wikipage import crawl_wikipedia_page, crawl_wikipedia_tool_2

BASE_1 = "data/cache/crawl_wikipedia"
BASE_2 = "data/cache/crawl_wikipedia_tool_2"
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
    out_path_1 = os.path.join(BASE_1, hashid_name)
    out_path_2 = os.path.join(BASE_2, hashid_name)

    # Only crawl if both output files do not exist
    if os.path.exists(out_path_1) and os.path.exists(out_path_2):
        continue

    if os.path.exists(out_path_1):
        data_crawl = load_json(out_path_1)
        if data_crawl is not None:
            html = data_crawl.get('crawl_wiki', {}).get('html_page', {}).get(link, None)
            if html is not None:
                title = normalize_title_multilang(html['title'])
                netloc = urlparse(link).netloc
                page_language = netloc.split('.')[0] if netloc.endswith('wikipedia.org') else ''

                # TOOL_1
                if not os.path.exists(out_path_1):
                    wiki_content_1 = crawl_wikipedia_page(title, page_language, link, base_delay=5)
                    out_1 = {
                        "crawl_wiki": {
                            "html_page": {
                                link: wiki_content_1
                            }
                        }
                    }
                    with open(out_path_1, "w", encoding="utf-8") as fout:
                        json.dump(out_1, fout, ensure_ascii=False, indent=2)
                    print("file TOOL_1: ", out_path_1)

                # TOOL_2
                if not os.path.exists(out_path_2):
                    wiki_content_2 = crawl_wikipedia_tool_2(title, page_language, link, base_delay=5)
                    out_2 = {
                        "crawl_wiki": {
                            "html_page": {
                                link: wiki_content_2
                            }
                        }
                    }
                    with open(out_path_2, "w", encoding="utf-8") as fout:
                        json.dump(out_2, fout, ensure_ascii=False, indent=2)
                    print("file TOOL_2: ", out_path_2)

                cnt += 1
                print("success: ", cnt)