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
import string
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentence_transformers import SentenceTransformer, util
from utils.norm_wikidata_pair import PROPERTY_MAPPINGS
from utils.crawl_sparql import get_last_oldid_before_date, send_sparql_request, find_all_qid, caculate_scores, fill_all_qid_search_sparql
model = SentenceTransformer("all-MiniLM-L6-v2")
import inspect

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/110.0.0.0 Safari/537.36"
    )
}

def match_relation_id(q_relation, top_k=3, retry_topk = 5):
    search_normalized = q_relation.lower().strip()
    
    if any(keyword in search_normalized for keyword in ['work', 'occupation', 'employer']):
        print(f"  Detected work question, using work heuristic")
        return {'P108': 'work at or employer', 'P101': 'field of work', 'P106': 'occupation'}
    
    if any(keyword in search_normalized for keyword in ['nationality', 'country', 'citizenship']):
        print(f"  Detected nationality question, using nationality heuristic")
        return {'P27': 'country of citizenship or nationality', 'P495': 'country of origin', 'P17': 'country'}

    # Exact match
    if search_normalized in PROPERTY_MAPPINGS:
        pid = PROPERTY_MAPPINGS[search_normalized]
        return {pid: search_normalized}

    # Semantic similarity
    keys = list(PROPERTY_MAPPINGS.keys())
    query_emb = model.encode(search_normalized, convert_to_tensor=True)
    key_embs = model.encode(keys, convert_to_tensor=True)
    
    cos_sim = util.cos_sim(query_emb, key_embs)[0].cpu().numpy()
    topk_idx = cos_sim.argsort()[-top_k:][::-1]
    topk_keys = [keys[i] for i in topk_idx]
    topk_scores = [cos_sim[i] for i in topk_idx]
    score_max = max(topk_scores) if topk_scores else 0

    # Nếu score_max < 0.5 thì lấy lại topk = 5
    if score_max < 0.5:
        top_k = retry_topk
        topk_idx = cos_sim.argsort()[-top_k:][::-1]
        topk_keys = [keys[i] for i in topk_idx]
        topk_scores = [cos_sim[i] for i in topk_idx]
        print(f"Score max < 0.5, lấy lại top-{top_k}")

    print(f"Top-{top_k} matches for '{q_relation}':")
    for i, (k, s) in enumerate(zip(topk_keys, topk_scores)):
        print(f"  {i+1}. {k} (score={s:.3f})")
    # Lọc trùng và trả về dict {pid: label}
    result = {}
    for k in topk_keys:
        pid = PROPERTY_MAPPINGS[k]
        if pid not in result:
            result[pid] = k
    return result

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
    
    print(f"Top {topk} QID có điểm cao nhất:")
    for i, item in enumerate(top_qids, 1):
        print(f"{i}. {item['label']} ({item['qid']}) - score={item['score']:.3f}")
    
    return top_qids

def request_sparql(dict_pid, qid_info):
    all_qid = " ".join([f"wd:{info['qid']}" for info in qid_info])
    all_pid = " ".join([f"wdt:{pid}" for pid in dict_pid.keys()])

    query = f"""
    SELECT
        ?item
        ?itemLabel
        ?itemDescription
        ?pLabel
        ?propertyValue
        ?propertyValueLabel
    WHERE {{
        VALUES ?item {{ {all_qid} }}
        VALUES ?p {{ {all_pid} }}

        ?item ?p ?propertyValue.

        SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en".
        }}
    }}
    """
    return send_sparql_request(query)


def _request_sparql(list_pid, qid_info):
    all_qid = ""
    for info in qid_info:
        all_qid += f"wd:{info['qid']} "
        
    all_pid = ""
    for item in list_pid:
        all_pid += f"wdt:{item} "
           
    query = f"""
        SELECT ?item ?itemLabel ?itemDescription ?propertyValue ?propertyValueLabel 
        WHERE {{
            VALUES ?item {{ {all_qid} }} 
            VALUES ?property {{ {all_pid} }} 
            ?item ?property ?propertyValue .
            
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "en".
            }}
        }}
        """
    print("Querry\n:", query)
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

def process_fillter_wikidata_api(q_relation, key_entity, get_history_old_id = 'False', dict_relation_id={}, limit=20):
    # Debug: print caller info to detect duplicated calls
    call_id = time.time()
    stack = inspect.stack()
    caller = stack[1]
    print(f"[DEBUG] process_fillter_wikidata_api called (id={call_id:.6f}) by {caller.function} in {os.path.basename(caller.filename)}:{caller.lineno}")

    key_entity = key_entity.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).strip()
    # qid_info = find_all_qid(key_entity, limit)
    qid_info, qid_description = fill_all_qid_search_sparql(key_entity, limit=limit)
    # print(qid_info)
    if not qid_info:
        print(f"No QID found for entity: {key_entity}")
        return None, None

    dict_result = {}
    all_context = qid_description
    
    result = request_sparql(dict_relation_id, qid_info)
    print(f"🔍 Found {len(result)} results with key_entity: {key_entity}")
    
    for item in result:
        id = item['item']['value'].split('/')[-1]
        label = item['itemLabel']['value'] 
        description = item.get('itemDescription', {}).get('value', '')
        prop_value = item['propertyValue']['value'].split('/')[-1] # q_id
        prop_value_label = item.get('propertyValueLabel', {}).get('value', '')
        relation_id = item['pLabel']['value'].split('/')[-1]
        relation_value = dict_relation_id[relation_id]
        
        if get_history_old_id == "False":
            prop_value_label_history = prop_value_label
        else:
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
        
        all_context += f"- {label} (description: {description}) - {relation_value}: {prop_value_label_history} (source: {prop_value})\n"
        
    return all_context, dict_result

# from src.generate_answer import gen_ans_multiple_choice
# def test():
#     question = "Who is the director of the film Days Of Betrayal?"
#     q_relation = "director of film 'Days Of Betrayal"
#     key_entity = "Days of Betrayal"
#     context = process_fillter_wikidata_api(q_relation, key_entity)
    
#     print(f"QID: {context}")
#     res = gen_ans_multiple_choice(question + f", {key_entity}", context, dataset='wiki', model="o3-mini")
#     print(f"Answer: {res}")
    
# if __name__ == "__main__":
#     question = "Who is the director of the film Days Of Betrayal?"
#     q_relation = "director of film 'Days Of Betrayal"
#     key_entity = "Days of Betrayal"
#     context = process_fillter_wikidata_api(q_relation, key_entity)
    
#     print(f"QID: {context}")
#     res = gen_ans_multiple_choice(question + f", {key_entity}", context, dataset='wiki', model="o3-mini")
#     print(f"Answer: {res}")