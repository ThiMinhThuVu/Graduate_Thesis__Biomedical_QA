import re
import os
import time
import requests
from utils.utils import load_json, save_json
from SPARQLWrapper import SPARQLWrapper, JSON
from utils.get_head_title_wikidata import get_wikidata_history_info

dict_relation = {
    "P22": "father",       
    "P25": "mother",       
    "P26": "spouse",       
    "P40": "child",       
    "P106": "occupation",  
    "P3373": "sibling"     
}

def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))
    
def remove_post_comma_titles(title):
    # Rule 1: Loại bỏ hậu tố sau dấu phẩy cuối cùng
    # return re.sub(r',.*', '', title).strip()
    return re.sub(r',[^,]*$', '', title).strip()

def remove_prefixes(title):
    prefixes = [
    'Princess', 'Sir', 'Lady', 'King', 'Queen', 'Prince', 'Lord', 'Baron', 'Baronet', 'Baronetess',
    'Duke', 'Count', 'Earl', 'Marquess', 'Viscount', 'Archduke', 'Elector', 'Palatine',
    'Emperor', 'Empress', 'Shah', 'Sultan', 'Tsar', 'Tsarina', 'Pharaoh',
    'Kaiser', 'Sheik', 'Professor', 'Admiral', 'General', 'Captain', 'Dr',
    'Reverend', 'Pope', 'Saint', 'Colonel', 'Grand Duke', 'Grand Duchess',
    'Infanta', 'Infante', 'Countess', 'Marquis', 'Marchioness', 'Viscountess',
    'Czar', 'Czarina', 'Dauphin', 'Major', 'Lieutenant', 'Commander',
    # Bổ sung honorifics và styles of address
    'The Honourable', 'Honourable',
    'The Right Honourable',
    'The Most Honourable',
    'The Most Noble',
    'The Worshipful',
    'The Venerable',
    'His Excellency', 'Her Excellency',
    'His Eminence', 'Her Eminence',
    'His Holiness', 'Her Holiness',
    'His Grace', 'Her Grace',
    'His Majesty', 'Her Majesty',
    'His Royal Highness', 'Her Royal Highness',
]

    
    for prefix in prefixes:
        pattern = fr'^{re.escape(prefix)}\s+'
        if re.search(pattern, title, re.IGNORECASE):
            return re.sub(pattern, '', title, flags=re.IGNORECASE).strip()
            
    return title

def remove_of_phrase(title):
    # Loại bỏ cụm "of ..." (vd: of Berg)
    return re.sub(r'\s+of\s+.*', '', title, flags=re.IGNORECASE).strip()

def search_sparql(title):
    endpoint_url = "https://query.wikidata.org/sparql"
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "MyWikidataSearchApp/1.0"
    }

    query_1 = f"""
    SELECT ?item ?itemLabel ?description
    WHERE {{
      ?item ?label "{title}"@en .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      OPTIONAL {{ ?item schema:description ?description FILTER (LANG(?description) = "en") }}
    }}
    """

    query_2 = f"""
    SELECT ?item ?itemLabel ?description WHERE {{
      {{
        ?item rdfs:label "{title}"@en .
      }}
      UNION
      {{
        ?item skos:altLabel "{title}"@en .
      }}
      
      SERVICE wikibase:label {{ 
        bd:serviceParam wikibase:language "en". 
      }}
      
      OPTIONAL {{ 
        ?item schema:description ?description 
        FILTER (LANG(?description) = "en") 
      }}
    }}
    """
    
    try:
        response_1 = requests.get(endpoint_url, params={"query": query_1}, headers=headers)
        response_1.raise_for_status()
        results = response_1.json()["results"]["bindings"]
        
        if results:
            return results
        
    except requests.exceptions.RequestException as e:
        print(f"Error executing query 1: {e}")
        return []
    except KeyError:
        print("Error parsing JSON for query 1.")
        return []
        
    try:
        response_2 = requests.get(endpoint_url, params={"query": query_2}, headers=headers)
        response_2.raise_for_status()
        results = response_2.json()["results"]["bindings"]
        
        if results:
            return results
            
    except requests.exceptions.RequestException as e:
        print(f"Error executing query 2: {e}")
        return []
    except KeyError:
        print("Error parsing JSON for query 2.")
        return []

    print("No results found after both searches.")
    return []

def extract_qid(url):
    return url.split('/')[-1]

def get_relevant_relation(item_id):
    """
    Truy vấn Wikidata để lấy các mối quan hệ và giá trị của một QID cụ thể.
    """
    qid = extract_qid(item_id)
    if not re.match(r"^Q\d+$", qid):
        print(f"'{qid}' là link wikipedia")
        return None
    
    endpoint_url = "https://query.wikidata.org/sparql"
    
    query = f"""
    SELECT ?relation ?relationLabel ?value ?valueLabel
    WHERE {{
        VALUES ?item {{ wd:{qid} }}

        VALUES ?prop {{
            wdt:P22     # cha
            wdt:P25     # mẹ
            wdt:P26     # vợ/chồng
            wdt:P40     # con
            wdt:P106    # nghề nghiệp
            wdt:P3373   # anh chị em
        }}

        OPTIONAL {{ ?item ?prop ?value . }}
        
        SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "en". 
            ?prop rdfs:label ?relationLabel.
            ?value rdfs:label ?valueLabel.
        }}
    }}
    """
    
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(endpoint_url, params={"query": query}, headers=headers)
    time.sleep(1)
    if response.status_code == 200:
        results = response.json()["results"]["bindings"]
        return results  
    else:
        raise Exception(f"Query failed with status {response.status_code}. Full response: {response.text}")
    
def check_relation_item(qid, q_relation, entity_1):
    list_relation = get_relevant_relation(qid)
    if list_relation == None:
        return False
    
    if "father" in q_relation.lower() or "mother" in q_relation.lower():
        relation = ["child"]
    elif "spouse" in q_relation.lower() or "wife" in q_relation.lower() or "husband" in q_relation.lower():
        relation = ["spouse"]
    elif "child" in q_relation.lower() or "son" in q_relation.lower() or "daughter" in q_relation.lower():
        relation = ["mother", "father"]
    elif "work" in q_relation.lower():
        relation = ["occupation"]
    else:
        return False

    for rel in list_relation:
        for expected_relation in relation:
            type_r = extract_qid(rel['relationLabel']['value'])
            relationLabel = dict_relation[type_r]
            if relationLabel == expected_relation:
                if 'valueLabel' in rel and rel['valueLabel']['value'].lower() == entity_1.lower():
                    return True
                    
    return False

def get_best_match_by_overlap(candidates, short_answer):
    """
    Chọn wikidata title có độ overlap token cao nhất với short_answer.
    """
    short_tokens = tokenize(short_answer)

    best_score = -1
    best_match = None

    for cand in candidates:
        cand_label = cand.get("itemLabel", {}).get("value", "")
        cand_tokens = tokenize(cand_label)
        if not cand_tokens:
            continue
        overlap = len(cand_tokens & short_tokens)
        union = len(cand_tokens | short_tokens)
        score = overlap / union if union else 0
        if score > best_score:
            best_score = score
            best_match = cand

    return best_match


def choose_answer(results, title, q_relation, entity_1): #q_relation là search_keyword của hop cuối
    """
    Chọn câu trả lời dựa trên số lượng và mối quan hệ.
    """
    # Trường hợp 1: Chỉ có 1 kết quả
    if len(results) == 1:
        qid = results[0]["item"]["value"]
        new_title = get_wikidata_history_info(qid)
        return new_title

    # Trường hợp 2: Có nhiều hơn 1 kết quả
    if len(results) > 1:
        found_matches = []
        for result in results:
            qid = result["item"]["value"]
            # Truy vấn mối quan hệ để xác nhận
            if check_relation_item(qid, q_relation, entity_1=entity_1):
                found_matches.append(result)

        if len(found_matches) == 1:
            qid = found_matches[0]["item"]["value"]
            new_title = get_wikidata_history_info(qid)
            return new_title

        if len(found_matches) > 1:
            # Tính toán độ trùng lặp/cosine với short answer
            best_match = get_best_match_by_overlap(found_matches, title)
            qid = best_match["item"]["value"]
            new_title = get_wikidata_history_info(qid)
            return new_title
            
    return title # Nếu k tìm thấy thì trả về cái này vì nó query trên sparql được

def normalize_title(full_title, q_relation, entity_1, date_string="2021-04-07"):
    """
    Chuẩn hóa một tiêu đề để tìm QID tương ứng trên Wikidata.
    """
    results = search_sparql(full_title)
    
    if results:
        final_answer = choose_answer(results, full_title, q_relation, entity_1)
        if final_answer: return final_answer
        
    # Bỏ tước vị và tìm lại
    # Rule 1: Loại bỏ hậu tố sau dấu phẩy cuối cùng
    title_rule1 = remove_post_comma_titles(full_title)
    if title_rule1 != full_title:
        results = search_sparql(title_rule1)
        if results:
            final_answer = choose_answer(results, title_rule1, q_relation, entity_1)
            if final_answer: return final_answer
    
    # Rule 2: Loại bỏ tiền tố
    title_rule2 = remove_prefixes(full_title)
    if title_rule2 != full_title:
        results = search_sparql(title_rule2)
        if results:
            final_answer = choose_answer(results, title_rule2, q_relation, entity_1)
            if final_answer: return final_answer
            
    # Rule 3: Loại bỏ "of ..."
    title_rule3 = remove_of_phrase(full_title)
    if title_rule3 != full_title:
        results = search_sparql(title_rule3)
        if results:
            final_answer = choose_answer(results, title_rule3, q_relation, entity_1)
            if final_answer: return final_answer

    return None

if __name__ == "__main__":
    type_q = "inference"
    folder = fr"E:\UETQuintet\wiki_{type_q}_results\all_ans"
    list_file = os.listdir(folder)
    
    for file in list_file:
        file_path = os.path.join(folder, file)
        sample = load_json(file_path)
        
        if "final_qi" in sample["1"]:
            if "normalized_ans" in sample["1"]["final_qi"]:
                continue
            
            entity_1 = sample["1"]["final_qi"]["key_entity"]
            q_relation = sample["1"]["final_qi"]["search_keyword"]
            short_ans = sample["1"]["final_qi"]["short_answer"]

            normalized_ans = normalize_title(short_ans, q_relation, entity_1, date_string="2021-04-07")
            sample["1"]["final_qi"]["normalized_ans"] = normalized_ans
            sample['final_answer']["normalized_ans"] = sample["1"]["final_qi"]["normalized_ans"]

            save_json(sample, file_path)
            print("wikidata: ", normalized_ans)
            print(file_path)
            time.sleep(2)