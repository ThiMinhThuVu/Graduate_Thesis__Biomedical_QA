import requests
import os
import json
import time
from typing import List, Set
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")
import torch

PROPERTY_MAPPINGS = {
    "award received": "P166",
    "cause of death": "P509",
    "child": "P40",
    "composer": "P86",
    "country": "P17",
    "country of citizenship": "P27",
    "country of origin": "P495",
    "creator": "P170",
    "date of birth": "P569",
    "date of death": "P570",
    "director": "P57",
    "doctoral advisor": "P184",
    "editor": "P98",
    "educated at": "P69",
    "employer": "P108",
    "father": "P22",
    "founded by": "P112",
    "has part": "P527",
    "inception": "P571",
    "manufacturer": "P176",
    "mother": "P25",
    "occupation": "P106",
    "performer": "P175",
    "place of birth": "P19",
    "place of burial": "P119",
    "place of death": "P20",
    "place of detention": "P2632",
    "presenter": "P371",
    "producer": "P162",
    "publication date": "P577",
    "publisher": "P123",
    "sibling": "P3373",
    "spouse": "P26",
    "student of": "P1066"
}

dict_p_relation = {
    "father": "P22",       
    "mother": "P25",
    "spouse": "P26",
    "child": "P40",
    "occupation": "P106",
    "sibling": "P3373"
}

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/110.0.0.0 Safari/537.36"
    )
}

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def dump_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
 
# Từ entity, query để thu về list các qid
def find_all_entity(entity_name, k=40):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity_name,
        "language": "en",
        "format": "json",
        "limit": k
    }

    response = requests.get(url, params=params, headers=headers, timeout=60)
    data_entity = response.json().get("search", [])
    if not data_entity:
        return []
    
    entities = []
    for item in data_entity:
        qid = item["id"]
        entities.append(qid)

    return entities

# Từ qid, thu về list các relation
def find_all_relation(qid):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetclaims",
        "entity": qid,
        "format": "json"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Lỗi khi truy vấn entity {qid}: {e}")
        return []

    claims = data.get("claims", {})
    return list(claims.keys())

# từ qid và rid, query ra answer
def get_answer_entity(qid, rid):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetclaims",
        "entity": qid,
        "format": "json"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Lỗi khi truy vấn entity {qid}: {e}")
        return []

    claims = data.get("claims", {})
    if rid not in claims:
        return []

    qids = []
    for claim in claims[rid]:
        mainsnak = claim.get("mainsnak", {})
        datavalue = mainsnak.get("datavalue", {})
        value = datavalue.get("value")

        if isinstance(value, dict) and "id" in value:
            qids.append(value["id"])
            return qids
        # evidence id trong compositional nếu không trả về qid sẽ trả về date info
        return ['date_information']

def filter_qids_by_properties(qids: List[str], property_ids: List[str], batch_size: int = 100, max_retries: int = 3) -> List[str]:
    """Filter QIDs that have any of the specified properties."""
    SPARQL_ENDPOINT = 'https://query.wikidata.org/sparql'
    USER_AGENT = "UETQuintet/1.0 (Research Project; Python/requests)"
    
    filtered_qids = []
    
    if not property_ids:
        print("No property IDs provided to filter by.")
        return []

    prop_values = " ".join([f"wdt:{pid}" for pid in property_ids])

    for i in range(0, len(qids), batch_size):
        batch = qids[i:i + batch_size]
        
        qid_values = " ".join([f"wd:{qid}" for qid in batch])

        # query = f"""
        # SELECT ?property ?propertyLabel ?value ?valueLabel ?valueDescription
        # WHERE {{
        #     VALUES ?entity {{ {qid_values} }}
        #     VALUES ?property {{ {prop_values} }}
            
        #     ?entity ?property ?value .
            
        #     SERVICE wikibase:label {{
        #         bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" .
        #     }}
        # }}
        # """
        query = f"""
        SELECT ?qid ?property ?propertyLabel ?value ?valueLabel ?valueDescription
        WHERE {{
            VALUES ?entity {{ {qid_values} }}
            VALUES ?property {{ {prop_values} }}
            
            BIND(?entity AS ?qid)
            ?entity ?property ?value .
            
            SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" .
            }}
        }}
        """

        headers = {
            'User-Agent': USER_AGENT,
            'Accept': 'application/sparql-results+json'
        }
        params = {'query': query, 'format': 'json'}

        for attempt in range(max_retries):
            try:
                response = requests.get(SPARQL_ENDPOINT, headers=headers, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()
                bindings = data.get('results', {}).get('bindings', [])
                for binding in bindings:
                    try:
                        qid = binding['qid']['value'].split('/')[-1]
                        p_value = binding['valueLabel']['value']
                        if p_value != "" or p_value is not None:
                            filtered_qids.append(qid)
                    except KeyError:
                        print(f"Warning: Unexpected binding format, skipping: {binding}")
                
                break

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Request failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Request failed after {max_retries} attempts: {e}")
                    break
    
    filtered_set = set(filtered_qids)
    ordered_filtered_qids = [qid for qid in qids if qid in filtered_set]
    return ordered_filtered_qids, bindings


def get_properties_for_relation(relation: str, question_text: str = "") -> List[str]:
    """Get property IDs for a given relation, handling work and nationality heuristics."""
    search_normalized = relation.lower().strip()
    
    if any(keyword in search_normalized for keyword in ['work', 'occupation', 'employer']):
        print(f"  Detected work question, using work heuristic")
        return ['P108', 'P101', 'P106']  # employer, field of work, occupation
    
    if any(keyword in search_normalized for keyword in ['nationality', 'country', 'citizenship']):
        print(f"  Detected nationality question, using nationality heuristic")
        return ['P27', 'P495', 'P17']  # citizenship, country of origin, country
    
    # Exact match
    if search_normalized in PROPERTY_MAPPINGS:
        return [PROPERTY_MAPPINGS[search_normalized]]

    # Semantic similarity
    keys = list(PROPERTY_MAPPINGS.keys())
    query_emb = model.encode(search_normalized, convert_to_tensor=True)
    key_embs = model.encode(keys, convert_to_tensor=True)
    
    cos_sim = util.cos_sim(query_emb, key_embs)[0]
    best_idx = torch.argmax(cos_sim).item()
    best_key = keys[best_idx]
   
    return [PROPERTY_MAPPINGS[best_key]]

def top_in_list(qid, entities_id):
    for index, entity_id in enumerate(entities_id):
        if entity_id == qid:
            return index + 1
    return None

if __name__ == "__main__":
    qtypes = ['compositional', 'inference']
    is_filter = True
    qtype = qtypes[0]
    version = 'compositional'
    file_path = f"data/wiki/dev/dev_{qtype}.json"
    cache_path = f"data/wiki/{'filter' if is_filter else 'unfilter'}/cache_{version}"
    os.makedirs(cache_path, exist_ok=True)

    threads = [0, 5236]
    # dev_compositional có 5236 câu
    # dev_inference có 1549 câu
    
    list_ID = [
    "05be9a0c0bde11eba7f7acde48001122",
    "20a713da0bde11eba7f7acde48001122",
    "20b818ba0bde11eba7f7acde48001122",
    "292b6e2e0bda11eba7f7acde48001122",
    "2982f22c0bdd11eba7f7acde48001122",
    "343506fc0bdc11eba7f7acde48001122",
    "35ecba540bdb11eba7f7acde48001122",
    "3e6de09c0bd911eba7f7acde48001122",
    "43828fba0bde11eba7f7acde48001122",
    "4b1b91720bde11eba7f7acde48001122",
    "5854d5380bd911eba7f7acde48001122",
  
]

    data = load_json(file_path)

    error_id = []
    for idx, query in tqdm(enumerate(data), desc="Processing: "):
        if not (threads[0] <= idx < threads[1]):
                continue

        id = query['_id']
        qtype = query['type']
        
        if id not in list_ID:
            continue

        cache_type = os.path.join(cache_path, f'{qtype}')
        os.makedirs(cache_type, exist_ok=True)

        result_file = os.path.join(cache_path, f'{qtype}/{id}.json')
        if os.path.exists(result_file):
            print(f"Đã xử lý câu id: {id}")
            continue
        
        print(f" id: {id}")
        results = {}
            
        # Lấy thông tin từ evidence
        evi = query['evidences']
        A = evi[0][0]
        R1 = evi[0][1]
        B_hop1 = evi[0][2]
        B_hop2 = evi[1][0] 
        R2 = evi[1][1]
        C = evi[1][2]

        # Bỏ qua các relation không có trong property mappings
        if R1 not in PROPERTY_MAPPINGS or R2 not in PROPERTY_MAPPINGS:
            error_id.append(query)
            print(f'Câu bị lỗi, tạm thời bỏ qua: {id}')
            continue
        
        # Lấy thông tin từ evidence id
        evi_id = query['evidences_id']
        A_id = evi_id[0][0]
        B_id = evi_id[0][2]
        C_id = evi_id[1][2]
        # Với dạng compositonal, entityC có thể là date_information hoặc qid
        # Với dạng inference, entityC đều là qid
        # Với 2 dạng comparison, evidences có thể rỗng

        # Kiểm tra xem khi qid của A có nằm trong list top k khi query A không
        results['A'] = A
        results['A_id'] = A_id
        results['all_qid_A'] = find_all_entity(A)
        results['topk_A'] = top_in_list(A_id, results['all_qid_A']) or 'Insufficient'

        # Kiểm tra R1 nằm trong top bao nhiêu với các relation thu được từ qid của A
        R1_id = PROPERTY_MAPPINGS[R1]
        results['R1'] = R1
        results['R1_id'] = R1_id
        results['A_all_relation'] = find_all_relation(A_id)
        results['topk_A_R1'] = top_in_list(R1_id, results['A_all_relation']) or 'Insufficient'
        
        # Kiểm tra từ A và R1 có ra được B không
        results['B_hop1'] = B_hop1
        results['B_hop2'] = B_hop2
        results['B_id'] = B_id

        results['qid_ans_B'] = get_answer_entity(A_id, R1_id)
        results['topk_A_R1_B'] = top_in_list(B_id, results['qid_ans_B']) or 'Insufficient'

        # Kiểm tra xem qid của B có nằm trong list top k khi query B không
        results['B_hop1'] = B_hop1
        results['B_hop2'] = B_hop2

        results['all_qid_B_hop1'] = find_all_entity(B_hop1)
        results['all_qid_B_hop2'] = find_all_entity(B_hop2)
        
        topk_B_hop1 = top_in_list(B_id, results['all_qid_B_hop1'])
        topk_B_hop2 = top_in_list(B_id, results['all_qid_B_hop2'])
        
        if topk_B_hop1 is None and topk_B_hop2 is None:
            results['topk_B'] = "Insufficient"
        elif topk_B_hop1 is None:
            results['topk_B'] = topk_B_hop2
        elif topk_B_hop2 is None:
            results['topk_B'] = topk_B_hop1
        else:
            results['topk_B'] = min(topk_B_hop1, topk_B_hop2)

        # Kiểm tra R2 nằm trong top bao nhiêu với các relation thu được từ qid của B
        R2_id = PROPERTY_MAPPINGS[R2]
        results['R2'] = R2
        results['R2_id'] = R2_id
        results['B_all_relation'] = find_all_relation(B_id)
        results['topk_B_R2'] = top_in_list(R2_id, results['B_all_relation']) or 'Insufficient'
        
        # Kiểm tra từ B và R2 có ra được C không
        results['C'] = C
        results['C_id'] = C_id
        results['qid_ans_C'] = get_answer_entity(B_id, R2_id)
        results['topk_B_R2_C'] = top_in_list(C_id, results['qid_ans_C']) or 'Insufficient'

        # Add QID filtering analysis
        print(f"  Adding QID filtering analysis for query {id}")
        
        # Get properties for R1 and R2
        r1_properties = get_properties_for_relation(R1, f"What is the {R1} of {A}?")
        r2_properties = get_properties_for_relation(R2, f"What is the {R2} of {B_hop1}?")
        
        # Filter QIDs for entity A based on R1 properties
        if results['all_qid_A']:
            if r1_properties:
                filtered_a_qids, bindings = filter_qids_by_properties(results['all_qid_A'], r1_properties)
                results['A_bindings'] = bindings
                results['filtered_qid_A'] = filtered_a_qids
                results['topk_A_filter'] = top_in_list(A_id, filtered_a_qids) or "Insufficient" 
            else:
                results['filtered_qid_A'] = results['all_qid_A']
                results['topk_A_filter'] = top_in_list(A_id, results['all_qid_A'])
        
        # Filter QIDs for entity B based on R2 properties
        topk_B_hop1_filter, topk_B_hop2_filter = None, None
        if results['all_qid_B_hop1']:
            if r2_properties:
                filtered_b_qids_hop1, bindings = filter_qids_by_properties(results['all_qid_B_hop1'], r2_properties)
                results['B_hop1_bindings'] = bindings
                results['filtered_qid_B_hop1'] = filtered_b_qids_hop1
                topk_B_hop1_filter = top_in_list(B_id, filtered_b_qids_hop1)
            else:
                results['filtered_qid_B_hop1'] = results['all_qid_B_hop1']
                topk_B_hop1_filter = top_in_list(B_id, results['all_qid_B_hop1'])
        
        if results['all_qid_B_hop2']:
            if r2_properties:
                filtered_b_qids_hop2, bindings = filter_qids_by_properties(results['all_qid_B_hop2'], r2_properties)
                results['B_hop2_bindings'] = bindings
                results['filtered_qid_B_hop2'] = filtered_b_qids_hop2
                topk_B_hop2_filter = top_in_list(B_id, filtered_b_qids_hop2) 
            else:
                results['filtered_qid_B_hop2'] = results['all_qid_B_hop2']
                topk_B_hop2_filter = top_in_list(B_id, results['all_qid_B_hop2'])
        
        if topk_B_hop1_filter is None and topk_B_hop2_filter is None:
            results['topk_B_filter'] = "Insufficient"
        elif topk_B_hop1_filter is None:
            results['topk_B_filter'] = topk_B_hop2_filter
        elif topk_B_hop2_filter is None:
            results['topk_B_filter'] = topk_B_hop1_filter
        else:
            results['topk_B_filter'] = min(topk_B_hop1_filter, topk_B_hop2_filter)

        dump_json(results, result_file)