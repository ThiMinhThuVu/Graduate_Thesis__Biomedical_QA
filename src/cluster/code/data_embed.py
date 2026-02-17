import os
import json

# =========================
# CONFIG
# =========================

SEARCH_TOPK = {
    "google": 20,
    "wiki": 10,
    "crawl_wiki": 3,
}

CACHE_HASHID_PATH = "data/cache/hash_id_IR_20.json"
CACHE_CRAWL_HASH_PATH = "data/cache/hash_id_crawl_wiki.json"
SEARCH_CACHE_FOLDER = "data/cache/IR_20"
CRAWL_FOLDER = "data/cache/crawl_wikipedia"
PRED_DIR = r"medhop_results_llm_create_mcq/all_ans"


# =========================
# LOAD
# =========================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

cache_index = load_json(CACHE_HASHID_PATH)
cache_crawl = load_json(CACHE_CRAWL_HASH_PATH)


# =========================
# UTILS
# =========================

def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def get_search_keyword_from_qidx(qidx):
    """
    Lấy search_keyword từ file prediction theo qidx
    """
    for fname in os.listdir(PRED_DIR):
        if fname.endswith(".json") and fname.split("_")[-1].replace(".json", "") == qidx:
            pred = load_json(os.path.join(PRED_DIR, fname))
            return pred["1"]["final_qi"]["search_keyword"]
    return None


def get_search_cache_data(search_kw):
    """
    Lấy file search cache từ keyword
    """
    query = normalize(search_kw)
    if query not in cache_index:
        return None

    cache_id = cache_index[query]
    search_cache_path = f"{SEARCH_CACHE_FOLDER}/{cache_id}"

    if not os.path.exists(search_cache_path):
        return None

    return load_json(search_cache_path)

def format_search_results(search_results, topk=5, search_type='GG'):
    context_parts = []
    for i, result in enumerate(search_results):            
        if i == topk:
            break
        snippet = f"-Reference: [{search_type}_{i+1}]: {result['title']} {result['snippet']}"
        
        context_parts.append(snippet)
                                       
    return "\n".join(context_parts), context_parts

def get_raw_wiki_context_from_articles(
    article_iterable,
    get_hash_id_func,
    get_link_func,
    get_title_func,
    cache_crawl_base,
):
    """
    Lấy raw context từ các bài wiki (không filter), trả về chuỗi context tổng hợp.
    """
    all_raw_context = ""
    for article in article_iterable:
        hash_id = get_hash_id_func(article)
        link = get_link_func(article)
        title = get_title_func(article)
        if not hash_id or not link:
            continue
        file_crawl = f"{cache_crawl_base}/{hash_id}"
        crawled_data = None
        if os.path.exists(file_crawl):
            crawl_wiki = load_json(file_crawl)
            crawled_data = crawl_wiki.get('crawl_wiki', {}).get('html_page', {}).get(link, None)
        if crawled_data and 'content' in crawled_data:
            raw_context = " ".join(crawled_data['content'])
            all_raw_context += f"\n[RAW_WIKI] {title}\n{raw_context}\n"
    return all_raw_context
# ==========================================================
# 1️⃣ LẤY CONTEXT TRƯỚC KHI LỌC (RAW)
# ==========================================================

def get_raw_context(qidx, source):
    """
    source: google | wiki | crawl_wiki
    return: raw context (chưa topK)
    """
    search_kw = get_search_keyword_from_qidx(qidx)
    if search_kw is None:
        return ""

    data = get_search_cache_data(search_kw)
    if data is None:
        return ""
    search_result = data['google']['search_results']
    context = ""

    # -------- GOOGLE --------
    if source == "google":
        
        context, list_article_ctx = format_search_results(search_result, topk=20, search_type='GG') 

    # -------- WIKI SNIPPET --------
    elif source == "wiki":
        context, list_article_ctx = format_search_results(search_result, topk=10, search_type='WIKI') 

    # -------- CRAWL WIKI --------
    elif source == "crawl_wiki":
        for item in data.get("wiki", {}).get("search_results", []):
            link = normalize(item.get("link", ""))

            if link not in cache_crawl:
                continue

            hashid = cache_crawl[link]
            crawl_path = f"{CRAWL_FOLDER}/{hashid}"

            if not os.path.exists(crawl_path):
                continue

            crawl_data = load_json(crawl_path)
            html = crawl_data.get("crawl_wiki", {}).get("html_page", {}).get(link, {})

            if html:
                content = html.get("content", [])
                context += " ".join(content) + "\n"

    return context.lower(), list_article_ctx


# ==========================================================
# 2️⃣ LẤY CONTEXT SAU KHI LỌC (TOPK)
# ==========================================================

def get_filtered_context(qidx, source):
    """
    source: google | wiki | crawl_wiki
    return: context sau khi áp dụng topK
    """
    search_kw = get_search_keyword_from_qidx(qidx)
    if search_kw is None:
        return ""

    data = get_search_cache_data(search_kw)
    if data is None:
        return ""

    context = ""
    topk = SEARCH_TOPK.get(source, None)

    if source == "google":
        for idx, item in enumerate(data.get("google", {}).get("search_results", [])):
            if idx >= topk:
                break
            context += f"{item.get('title','')} {item.get('snippet','')}\n"

    elif source == "wiki":
        for idx, item in enumerate(data.get("wiki", {}).get("search_results", [])):
            if idx >= topk:
                break
            context += f"{item.get('title','')} {item.get('snippet','')}\n"

    elif source == "crawl_wiki":
        idx_crawl = 0
        for item in data.get("wiki", {}).get("search_results", []):
            if idx_crawl >= topk:
                break

            link = normalize(item.get("link", ""))

            if link not in cache_crawl:
                continue

            hashid = cache_crawl[link]
            crawl_path = f"{CRAWL_FOLDER}/{hashid}"

            if not os.path.exists(crawl_path):
                continue

            crawl_data = load_json(crawl_path)
            html = crawl_data.get("crawl_wiki", {}).get("html_page", {}).get(link, {})

            if html:
                content = html.get("content", [])
                context += " ".join(content) + "\n"
                idx_crawl += 1

    return context.lower()

from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import os 

device = "cpu"

embedding_model = SentenceTransformer(
    "pritamdeka/S-BioBert-snli-multinli-stsb",
    device=device
)

def get_embedding(text):
    embedding = embedding_model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True  
    )
    return embedding


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
        
if __name__ == "__main__":
    embed_file = 'E:\MAS_UETQuintet\data\cache\embed_ctx_merge_google_wiki_YN.json'
    metadata = load_json(r'data\metadata_medhop.json')
    dict_output = {}
    
    for idx, (qidx, item) in enumerate(metadata.items()):    
        
        if item["target_answer_type"] != 'correctness' or item['type_q'] != 'direct': #YN
            continue
        list_embed_article = []
        context, list_article_gg = get_raw_context(qidx, 'google')
        context, list_article_wiki = get_raw_context(qidx, 'wiki')
        list_article = list_article_gg + list_article_wiki
        
        for article in list_article:
            embed_article = get_embedding(article)
            # Đảm bảo chuyển np.ndarray thành list
            if isinstance(embed_article, np.ndarray):
                embed_article = embed_article.tolist()
            list_embed_article.append(embed_article)
            
        dict_output[qidx] = {
            'docs': list_article,
            'embed_docs': list_embed_article
        }
        
        
        print(idx, embed_file)
        
    save_json(dict_output, embed_file)