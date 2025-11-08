import re
import requests
import time
import sys
import os
from bs4 import BeautifulSoup
from thefuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer, util
from utils.norm_wikidata_pair import PROPERTY_MAPPINGS
from utils.crawl_sparql import get_last_oldid_before_date, send_sparql_request, find_all_qid, caculate_scores
model = SentenceTransformer("all-MiniLM-L6-v2")

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/110.0.0.0 Safari/537.36"
    )
}

def match_relation_id(q_relation):
    search_normalized = q_relation.lower().strip()
    
    if any(keyword in search_normalized for keyword in ['work', 'occupation', 'employer']):
        print(f"  Detected work question, using work heuristic")
        return ['P108', 'P101', 'P106'], ['work at or employer', 'field of work', 'occupation']  # employer, field of work, occupation
    
    if any(keyword in search_normalized for keyword in ['nationality', 'country', 'citizenship']):
        print(f"  Detected nationality question, using nationality heuristic")
        return ['P27', 'P495', 'P17'], ['country of citizenship or nationality', 'country of origin', 'country']  # citizenship, country of origin, country

    
    # Exact match
    if search_normalized in PROPERTY_MAPPINGS:
        return [PROPERTY_MAPPINGS[search_normalized]], [search_normalized]

    # Semantic similarity
    keys = list(PROPERTY_MAPPINGS.keys())
    query_emb = model.encode(search_normalized, convert_to_tensor=True)
    key_embs = model.encode(keys, convert_to_tensor=True)
    
    cos_sim = util.cos_sim(query_emb, key_embs)[0]
    best_idx = torch.argmax(cos_sim).item()
    best_key = keys[best_idx]
    best_score = cos_sim[best_idx].item()
    
    print(f"Best match for '{q_relation}' → '{best_key}' (score={best_score:.3f})")
    
    return [PROPERTY_MAPPINGS[best_key]], [best_key]
    

def get_best_match_qid(qid_info, key_entity_relation, topk=20):
    scored_qids = []
    
    print(f"🔍 Tổng số QID tìm thấy: {len(qid_info)}")

    for info in qid_info:
        qid = info.get("qid")
        label = info.get("label", "")
        description = info.get("description", "")
        text = f"{label} {description}".strip()
        
        scores = caculate_scores(text, key_entity_relation)
        final_score = scores["cosine"]
        
        scored_qids.append({
            "qid": qid,
            "label": label,
            "description": description,
            "score": final_score
        })
    
    scored_qids = sorted(scored_qids, key=lambda x: x["score"], reverse=True)
    
    top_qids = scored_qids[:topk]
    
    print(f"✅ Top {topk} QID có điểm cao nhất:")
    for i, item in enumerate(top_qids, 1):
        print(f"{i}. {item['label']} ({item['qid']}) - score={item['score']:.3f}")
    
    return top_qids

def request_sparql(pid, qid_info):
    all_qid = ""
    for info in qid_info:
        all_qid += f"wd:{info['qid']} "
        
    query = f"""
        SELECT ?item ?itemLabel ?itemDescription ?propertyValue ?propertyValueLabel 
        WHERE {{
            VALUES ?item {{ {all_qid} }}
            ?item wdt:{pid} ?propertyValue.    
            
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "en".
            }}
        }}
        """
    result = send_sparql_request(query)

    return result

def get_pid_value(qid, pid):
    oldid = get_last_oldid_before_date(qid)
    if oldid:
        print(f"Oldid cuối cùng: {oldid}")
        url = f"https://www.wikidata.org/w/index.php?title={qid}&oldid={oldid}"
    else:
        url = f"https://www.wikidata.org/wiki/{qid}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    
    group = soup.find("div", {"id": pid})
    values = []
    if group:
        mainsnaks = group.find_all("div", class_="wikibase-statementview-mainsnak-container")
        for snak in mainsnaks:
            val_div = snak.find("div", class_="wikibase-snakview-value")
            if val_div:
                values.append(val_div.get_text(strip=True))
            
    return values

def process_fillter_wikidata_api(q_relation, key_entity, limit=20):
    qid_info = find_all_qid(key_entity, limit)
    print(qid_info)
    print(f"🔍 Found {len(qid_info)} QID for entity: {key_entity}")
    if not qid_info:
        print(f"No QID found for entity: {key_entity}")
        return None

    list_relation_id, list_relation_value = match_relation_id(q_relation) # trả về [], []
    dict_result = {}
    all_context = ""
    
    for relation_id, relation_value in zip(list_relation_id, list_relation_value):
        result = request_sparql(relation_id, qid_info)
        print(f"🔍 Found {len(result)} results")
        
        for idx, item in enumerate(result):
            id = item['item']['value'].split('/')[-1]
            label = item['itemLabel']['value'] 
            description = item.get('itemDescription', {}).get('value', '')
            prop_value = item['propertyValue']['value'].split('/')[-1]
            prop_value_label = item.get('propertyValueLabel', {}).get('value', '')

            prop_value_label_history = get_pid_value(id, relation_id)
            
            if id in dict_result:
                dict_result[id].append({
                    "label": label,
                    "description": description,
                    "property_value": prop_value,
                    "property_value_label": prop_value_label,
                    "prop_value_label_history": prop_value_label_history
                })
            else:
                dict_result[id] = [{
                    "label": label,
                    "description": description,
                    "property_value": prop_value,
                    "property_value_label": prop_value_label,
                    "prop_value_label_history": prop_value_label_history
                }]
            
            select = chr(idx + 65)
            all_context += f"{select}. {label} (description: {description}) - {relation_value}: {prop_value_label_history} (source: {prop_value})\n"
    return all_context, dict_result

from src.generate_answer import gen_ans_multiple_choice
if __name__ == "__main__":
    question = "What is the place of birth of that performer?"
    q_relation = "place of birth"
    key_entity = "When the Stars Go Blue"
    context = process_fillter_wikidata_api(q_relation, key_entity)
    
    print(f"QID: {context}")
    res = gen_ans_multiple_choice(question + f", {key_entity}", context, dataset='wiki', model="o3-mini")
    print(f"Answer: {res}")