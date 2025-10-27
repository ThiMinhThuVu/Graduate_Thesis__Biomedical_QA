import re
import requests
import time
import sys
import os
from thefuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.util import *
from src.retrieval import *
model = SentenceTransformer("all-MiniLM-L6-v2")

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/110.0.0.0 Safari/537.36"
    )
}

def send_sparql_request(query, max_retries=5):
    endpoint_url = "https://query.wikidata.org/sparql"
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "UETQuintetBot/1.0"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(endpoint_url, params={"query": query}, headers=headers)
            if response.status_code == 200:
                return response.json()["results"]["bindings"]
            elif response.status_code == 429:
                wait = 2 ** attempt
                print(f"⚠️ 429 - rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"❌ HTTP {response.status_code}")
                break
        except Exception as e:
            print(f"❌ Connection error: {e}")
            time.sleep(2 ** attempt)
    return []

def caculate_scores(search, golden):
    if search.lower() == golden.lower():
        return {'cosine': 1.0, 'fuzz': 1.0}

    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
    try:
        tfidf_matrix = vectorizer.fit_transform([search, golden])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except ValueError:
        cosine_sim = 0.0

    score_fuzz = fuzz.token_set_ratio(search, golden) / 100.0
    return {'cosine': cosine_sim, 'fuzz': score_fuzz}

def normalize_text(text):
    text = re.sub(r'\s*\([^)]*\)\s*', '', text)
    return text.strip()

def norm_key_entity(search_kw, cos_threshold=0.7, fuzz_threshold=0.7):
    api_url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": search_kw,
        "format": "json"
    }

    try:
        response = requests.get(api_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi gọi API: {e}")
        return None

    search_results = data.get("query", {}).get("search", [])
    if not search_results:
        print(f"Không tìm thấy kết quả cho '{search_kw}'.")
        return None

    entities = [normalize_text(res["title"]) for res in search_results]

    scored_entities = []
    for ent in entities:
        scores = caculate_scores(search_kw, ent)
        avg_score = (scores["cosine"] + scores["fuzz"]) / 2
        scored_entities.append({
            "entity": ent,
            "cosine": scores["cosine"],
            "fuzz": scores["fuzz"],
            "avg": avg_score
        })

    scored_entities.sort(key=lambda x: x["avg"], reverse=True)
    best = scored_entities[0]

    if best["cosine"] < cos_threshold and best["fuzz"] < fuzz_threshold:
        return {"error": f"Không có entity nào đủ giống '{search_kw}'."}
    
    print(f'Các entity: {entities} -> Chọn "{best["entity"]}"')
    return best["entity"]

def extract_qid(url):
    return url.split("/")[-1]

def find_all_qid(key_entity):
    search_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": key_entity,
        "language": "en",
        "format": "json"
    }
    try:
        resp = requests.get(search_url, headers=headers, params=params)
        resp.raise_for_status()
        search_data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Lỗi khi tìm QID cho {key_entity}: {e}")
        return None

    if not search_data.get("search"):
        print(f"⚠️ Không tìm thấy QID cho entity: {key_entity}")
        return None

    qid_info = []
    for item in search_data["search"]:
        qid = item.get("id")
        label = item.get("label", "")
        qid_info.append({
            "qid": qid,
            "label": label
        })

    return qid_info

def best_match_qid(qid_info, key_entity):
    # print(f"🔍 Các qid: {qid_info}")
    best_score = -1
    best_qid = None
    
    print(f"🔍 Các qid: {qid_info}")
    for info in qid_info:
        qid = info["qid"]
        label = info["label"]
        scores = caculate_scores(label, key_entity)

        final_score = (scores["cosine"] + scores["fuzz"]) / 2
        if final_score > best_score:
            best_score = final_score
            best_qid = qid

    print(f"✅ Chọn QID tốt nhất: {best_qid}")
    return best_qid

def find_sparql_context(entity):
    key_entity = norm_key_entity(entity)
    if not key_entity or isinstance(key_entity, dict) and "error" in key_entity:
        print(f"❌ Không chuẩn hóa được entity: {entity}")
        key_entity = entity

    qids = find_all_qid(key_entity)
    if not qids:
        return ""

    qid_value = best_match_qid(qids, key_entity)
    if not qid_value.startswith("wd:"):
        qid_value = f"wd:{qid_value}"
    
    sparql_context = []

    query = f"""
    SELECT ?property ?propertyLabel ?valueLabel ?valueDescription
    WHERE {{
        VALUES ?entity {{ {qid_value} }}

        ?entity ?p ?statement .
        ?property wikibase:claim ?p .

        ?statement ?ps ?value .
        ?property wikibase:statementProperty ?ps .

        SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" .
        }}
    }}
    """

    result = send_sparql_request(query)

    for item in result:
        property_label = item.get("propertyLabel", {}).get("value", "")
        value_label = item.get("valueLabel", {}).get("value", "")
        value_description = item.get("valueDescription", {}).get("value", "")
        sparql_context.append(property_label + " " + value_label + ", description: " + value_description)

    return sparql_context 

def select_top_documents(key_entity, ranked_docs, max_tokens=300):
    selected = []
    total_tokens = 0

    for item in ranked_docs:
        text = item['text']
        num_tokens = len(text.split())

        if total_tokens + num_tokens > max_tokens:
            break

        selected.append(text)
        total_tokens += num_tokens
    
    return key_entity + ": " + "; ".join(selected)

if __name__ == "__main__":
    # id = "ef32ebf40bdc11eba7f7acde48001122"
    result = find_sparql_context("Bigas Luna")
    ranked_docs = find_most_relevant_doc("place of death", result)
    print(select_top_documents("Bigas Luna", ranked_docs))