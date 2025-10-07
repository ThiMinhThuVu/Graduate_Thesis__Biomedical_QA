import re
import os
import time
import requests
from utils.util import load_json, save_json

dict_relation = {
    "P22": "father",       
    "P25": "mother",       
    "P26": "spouse",       
    "P40": "child",       
    "P106": "occupation",  
    "P3373": "sibling"     
}

dict_p_relation = {
    "father": "P22",       
    "mother": "P25",
    "spouse": "P26",
    "child": "P40",
    "occupation": "P106",
    "sibling": "P3373"
}

def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))
    

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
        
        res_output = []
        for res in results:
            if 'entity/Q' in res['item']['value']:
                res_output.append(res)
                
        if res_output:
            return res_output

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
        
        res_output = []
        for res in results:
            if 'entity/Q' in res['item']['value']:
                res_output.append(res)
                
        if res_output:
            return res_output
        
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

def sparql_find_parent(qid, q_relation):
    if "father" in q_relation.lower():
        q_relation = "father"
    elif "mother" in q_relation.lower():
        q_relation = "mother"
    elif "spouse" in q_relation.lower() or "wife" in q_relation.lower() or "husband" in q_relation.lower():
        q_relation = "spouse"
    elif "child" in q_relation.lower() or "son" in q_relation.lower() or "daughter" in q_relation.lower():
        q_relation = "child"
    elif "sibling" in q_relation.lower() or "brother" in q_relation.lower() or "sister" in q_relation.lower():
        q_relation = "sibling"
    elif "work" in q_relation.lower():
        q_relation = "occupation"
    else:
        return False
    
    p_relation = dict_p_relation.get(q_relation.lower())
    print("p_relation: ", p_relation)
    if not p_relation:
        return None
    
    
    
    endpoint_url = "https://query.wikidata.org/sparql"
    
    query = f"""
    SELECT ?property ?propertyLabel
    WHERE {{
        wd:{qid} wdt:{p_relation} ?property .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(endpoint_url, params={"query": query}, headers=headers)
    # time.sleep(1)
    if response.status_code == 200:
        results = response.json()["results"]["bindings"]
        return results  
    else:
        raise Exception(f"Query failed with status {response.status_code}. Full response: {response.text}")


def sparql_find_child(item_id, q_relation):
    if "father" in q_relation.lower() or "mother" in q_relation.lower():
        relation = ["child"]
    elif "spouse" in q_relation.lower() or "wife" in q_relation.lower() or "husband" in q_relation.lower():
        relation = ["spouse"]
    elif "child" in q_relation.lower() or "son" in q_relation.lower() or "daughter" in q_relation.lower():
        relation = ["mother", "father"]
    elif "sibling" in q_relation.lower() or "brother" in q_relation.lower() or "sister" in q_relation.lower():
        relation = ["sibling"]
    elif "work" in q_relation.lower():
        relation = ["occupation"]
    else:
        return False
    
    list_results = []
    for r in relation:
        q_relation = r        
        p_relation = dict_p_relation.get(q_relation.lower())
        print("p_relation: ", p_relation)
        if not p_relation:
            return None
        
        qid = extract_qid(item_id)
        if not re.match(r"^Q\d+$", qid):
            print(f"'{qid}' là link wikipedia")
            return None
        
        endpoint_url = "https://query.wikidata.org/sparql"
        
        query = f"""
        SELECT ?property ?propertyLabel
        WHERE {{
            wd:{qid} wdt:{p_relation} ?property .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """
        
        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(endpoint_url, params={"query": query}, headers=headers)
        # time.sleep(1)
        if response.status_code == 200:
            results = response.json()["results"]["bindings"]
            list_results.extend(results)
        else:
            raise Exception(f"Query failed with status {response.status_code}. Full response: {response.text}")
    return list_results

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


def get_nom_qid(sample, date_string="2021-04-07"):
    """
    Chuẩn hóa một tiêu đề để tìm QID tương ứng trên Wikidata.
    """
    full_name = sample["1"]["final_qi"]["short_answer"]
    prefix = sample["1"]["final_qi"]['prefix']
    name = sample["1"]["final_qi"]['name']
    suffix = sample["1"]["final_qi"]['suffix']
 
    list_qid = []
    #KB1: chạy full name
    results = search_sparql(full_name)
    
    if results:
        list_qid.extend(results) 
    
    #Kb2: chỉ name
    if name != full_name:
        results = search_sparql(name)
        if results:
            list_qid.extend(results) 

    #kb3: chỉ prefix + name
    if prefix != "NAN":
        name_tmp = prefix + " " + name
        if name_tmp != full_name:
            results = search_sparql(name_tmp)
            if results:
                list_qid.extend(results) 

    #kb4: chỉ name + suffix
    if suffix != "NAN":
        name_tmp = name + " " + suffix
        ok = 0
        if name_tmp != full_name:
            results = search_sparql(name_tmp)
            if results:
                ok = 1
                list_qid.extend(results)

        if ok == 0:
            name_tmp = name + ", " + suffix
            if name_tmp != full_name:
                results = search_sparql(name_tmp)
                if results:
                    list_qid.extend(results) 

    return list_qid

def norm_key_entity(entity, data):
    match = re.match(r"^(.*?):", data)
    if match:
        result = match.group(1)
        entity = result
    
    return entity

def verify_relations(dict_output):
    verify_list = []

    # duyệt qua từng parent và danh sách con của nó
    for parent_qid, children in dict_output.get("parent", {}).items():
        for child_entry in children:
            child_qid = child_entry["property"]["value"].split("/")[-1]  # lấy QID của child
            
            # kiểm tra trong child có chứa parent ngược lại không
            if child_qid in dict_output.get("child", {}):
                for possible_parent in dict_output["child"][child_qid]:
                    qid_back = possible_parent["property"]["value"].split("/")[-1]
                    
                    # nếu có liên kết ngược (child -> parent)
                    if qid_back == parent_qid:
                        verify_list.append({
                            "parent_qid": child_qid,
                            "parent_label": child_entry["propertyLabel"]["value"] if "propertyLabel" in child_entry else None,
                            "child_qid": parent_qid,
                            "child_label": possible_parent["propertyLabel"]["value"] if "propertyLabel" in possible_parent else None
                        })

    # thêm vào dict_output
    dict_output["verify"] = verify_list
    return dict_output


if __name__ == "__main__":
    type_q = "inference"
    folder = fr"E:\UETQuintet\wiki_{type_q}_results\all_ans"
    # golden_data = load_json(r'data\wiki\dev\dev_inference.json')
    list_file = os.listdir(folder)
    
    for file in list_file:
        file_path = os.path.join(folder, file)
        sample = load_json(file_path)
        if sample["QIDX"] != "054fb79c0baf11ebab90acde48001122": #0bad718c0bb011ebab90acde48001122
            continue
        
        if "final_qi" in sample["1"]:
            # if "normalized_ans" in sample["1"]["final_qi"]:
            #     continue
            key_entity = sample["1"]["final_qi"]["key_entity"]
            q_relation = sample["1"]["final_qi"]["search_keyword"]
            short_ans = sample["1"]["final_qi"]["short_answer"]
            retrieval_inf = sample["1"]["final_qi"]["retrieval_information"]
            print("key_entity: ", key_entity)
            dict_output = {"parent": {}, "child": {}}
            
            key_entity = norm_key_entity(key_entity, retrieval_inf)
            print("norm key_entity: ", key_entity)
            #find parent
            list_parent = []
            list_qid = search_sparql(key_entity)
            
            # print(list_qid)
            check_qid_parent = set()
            for item in list_qid:
                qid = extract_qid(item["item"]["value"])
                if not re.match(r"^Q\d+$", qid):
                    print(f"'{qid}' là link wikipedia")
                    continue
                else:
                    if qid in check_qid_parent:
                        continue
                    check_qid_parent.add(qid)

                parent = sparql_find_parent(qid, q_relation)
                list_parent.extend(parent)
                dict_output["parent"][qid] = parent
                
            # print("list_parent: ", list_parent)
                
            #find child
            list_child = []
            list_qid = get_nom_qid(sample)
            check_qid_child = set()
            # print(list_qid)
            for item in list_qid:
                qid = extract_qid(item["item"]["value"])
                if not re.match(r"^Q\d+$", qid):
                    print(f"'{qid}' là link wikipedia")
                    continue
                else:
                    if qid in check_qid_child:
                        continue
                    check_qid_child.add(qid)

                child = sparql_find_child(qid, q_relation)
                list_child.extend(child)
                dict_output["child"][qid] = child

            # print("list_child: ", list_child)
            dict_output = verify_relations(dict_output)
            print("dict_output: ", dict_output)