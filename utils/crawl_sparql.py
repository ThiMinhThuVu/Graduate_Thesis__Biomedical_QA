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

from src.retrieval import *
model = SentenceTransformer("all-MiniLM-L6-v2")

PROPERTY_MAPPINGS = {
    "place of birth": "P19",
    "birth": "P19",
    "place of death": "P20",
    "death": "P20",
    "place of burial": "P119",
    "burial": "P119",
    "graduation": "P69",
    "educated at": "P69",
    "education": "P69",
    "studied": "P69",
    "country of origin": "P495",
    "country of citizenship": "P27",
    "country of nationality": "P27",
    "nationality": "P27",
    "country": "P17",
    "occupation": "P106",
    "work at": "P108",
    "workplace": "P108",
    "work place": "P108",
    "work location": "P108",
    "employer": "P108",
    "field of work": "P101",
    "spouse of": "P26",
    "spouse": "P26",
    "partner": "P26",
    "married": "P26",
    "child of": "P40",
    "child": "P40",
    "children": "P40",
    "parent": "P22",
    "mother": "P25",
    "father": "P22",
    "sibling": "P3373",
    "award received": "P166",
    "award": "P166",
    "prize": "P166",
    "position held": "P39",
    "position": "P39",
    "office": "P39",
    "member of": "P463",
    "member": "P463",
    "affiliation": "P463",
    "residence": "P551",
    "located": "P276",
    "location": "P276",
    "headquarters location": "P159",
    "headquarter": "P159",
    "founded": "P571",
    "inception": "P571",
    "cast member": "P161",
    "cast": "P161",
    "genre": "P136",
    "date of birth": "P569",
    "date of death": "P570",
    "date of foundation": "P571",
    'place of study': "P69",
    "study location": "P69",
    "educated at": "P69",
    "study": "P69",
    "birthday": "P569",
    "place of detention": "P2632",
    "prison": "P2632",
    "cause of death": "P509",
    "founder of": "P112",
    "director": "P57",
    "publication date": "P577",
    "producer": "P162",
}

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

def norm_key_entity(search_kw, lang="en", cos_threshold=0.4, fuzz_threshold=0.4):
    api_url = f"https://{lang}.wikipedia.org/w/api.php"

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
    
    print(f'Các entity: {entities} -> Chọn "{best["entity"]}" với điểm cosine {best["cosine"]:.4f} và fuzz {best["fuzz"]:.4f}')
    return best["entity"]

def extract_qid(url):
    return url.split("/")[-1]

def find_all_qid(key_entity, lang="en"):
    search_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": key_entity,
        "language": lang,
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
        description = item.get("description", "")
        qid_info.append({
            "qid": qid,
            "label": label,
            "description": description
        })

    return qid_info

def best_match_qid(qid_info, key_entity, lang="en"):
    # print(f"🔍 Các qid: {qid_info}")
    best_score = -1
    best_qid = None
    
    print(f"🔍 Các qid: {qid_info}")
    for info in qid_info:
        qid = info["qid"]
        label = info["label"]
        description = info["description"]
        scores = caculate_scores(label, key_entity)

        final_score = (scores["cosine"] + scores["fuzz"]) / 2
        if final_score > best_score:
            best_score = final_score
            best_qid = qid

    print(f"✅ Chọn QID tốt nhất: {best_qid} với điểm số {best_score:.4f}")
    return best_qid

def find_sparql_context(entity, search_kw, lang="en"):
    key_entity = norm_key_entity(entity, lang=lang)
    if not key_entity or isinstance(key_entity, dict) and "error" in key_entity:
        print(f"❌ Không chuẩn hóa được entity: {entity}")
        key_entity = entity

    qids = find_all_qid(key_entity, lang=lang)
    if not qids:
        return ""

    qid_value = best_match_qid(qids, key_entity, lang=lang)
    if not qid_value.startswith("wd:"):
        qid_value = f"wd:{qid_value}"
    
    sparql_context = []

    relation, relation_id = None, None
    for rel_text, rel_id in PROPERTY_MAPPINGS.items():
        if rel_text in search_kw.lower():
            relation = rel_text
            relation_id = rel_id
            break
    if not relation:
        print(f"⚠️ Không tìm thấy mapping cho câu hỏi")
        return ""

    print(f"🔍 Tìm quan hệ '{relation}' với ID '{relation_id}' cho {key_entity}: {qid_value}")
    query = f"""
    SELECT ?item ?itemLabel ?itemDescription WHERE {{
        VALUES ?entity {{ {qid_value} }}
        ?entity wdt:{relation_id} ?item .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
    }}
    """

    result = send_sparql_request(query)
    
    for row in result:
        item_uri = row["item"]["value"]
        item_qid = extract_qid(item_uri)
        title = row.get("itemLabel", {}).get("value", "")
        desc = row.get("itemDescription", {}).get("value", "")

        if desc == "":
            continue

        sparql_context.append({
            "relation": relation,
            "qid": item_qid,
            "title": title,
            "description": desc
        })

    # 5️⃣ Kiểm tra kết quả cuối
    if not sparql_context:
        print(f"⚠️ Không có kết quả SPARQL nào cho {key_entity}: {qid_value}")
        return ""
    else:
        print(f"✅ Tìm thấy {len(sparql_context)} kết quả cho {key_entity}")
        return key_entity + ": " + relation + " " + sparql_context[0]['title'] + ", " + sparql_context[0]['description']

if __name__ == "__main__":
    # id = "ef32ebf40bdc11eba7f7acde48001122"
    # result = find_sparql_context("George Dewhurst", "place of death George Dewhurst", "Where did that director die?", lang="en")
    result = find_sparql_context("Bigas Luna", "place of death", "Where did that director die?")
    print(result)