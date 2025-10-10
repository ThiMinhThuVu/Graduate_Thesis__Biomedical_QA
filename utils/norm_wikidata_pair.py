import re
import os
import time
import requests


PROPERTY_MAPPINGS = {
    "birth": "P19",
    "place of birth": "P19",
    "death": "P20",
    "place of death": "P20",
    "burial": "P119",
    "place of burial": "P119",
    "graduation": "P69",
    "education": "P69",
    "educated at": "P69",
    "studied": "P69",
    "nationality": "P27",
    "country of citizenship": "P27",
    "citizen": "P27",
    "occupation": "P106",
    "work": "P106",
    "employer": "P108",
    "spouse": "P26",
    "partner": "P26",
    "married": "P26",
    "child": "P40",
    "children": "P40",
    "parent": "P22",
    "mother": "P25",
    "father": "P22",
    "sibling": "P3373",
    "award": "P166",
    "award received": "P166",
    "prize": "P166",
    "position": "P39",
    "position held": "P39",
    "office": "P39",
    "member": "P463",
    "member of": "P463",
    "affiliation": "P463",
    "residence": "P551",
    "located": "P276",
    "location": "P276",
    "headquarter": "P159",
    "headquarters location": "P159",
    "founded": "P571",
    "inception": "P571",
    "creator": "P170",
    "author": "P50",
    "director": "P57",
    "producer": "P162",
    "cast": "P161",
    "cast member": "P161",
    "genre": "P136",
}

dict_p_relation = {
    "father": "P22",       
    "mother": "P25",
    "spouse": "P26",
    "child": "P40",
    "occupation": "P106",
    "sibling": "P3373"
}

# Bidirectional relationship properties (require verification)
BIDIRECTIONAL_PROPERTIES = set(dict_p_relation.values())

API_KEY_GG = 'Bearer 5C776EC6-9757-45BF-B13E-85F3C4860C18'

def search_google(query):
    query += " site:wikidata.org"
    
    max_retries = 30
    retry_delay = 30
   
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'https://api.search1api.com/search',
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': API_KEY_GG 
                },
                json={
                    "query": query,
                    "search_service": "google",
                    "max_results": 10,
                    "crawl_results": 0,
                    "image": False,
                    "language": "en"
                },
                timeout=30
            )

            if response.status_code in [200, 404]:
                return response.json().get("results", [])
            else:
                raise RuntimeError(f"Search API failed with status code {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise e 
            
def norm_keyentity_wikidata_api(results, topk=1):   
    if not results:
        return []  

    filtered = []

    for idx, r in enumerate(results[0:topk]):
        if "wikidata.org/wiki/" in r['link']:
            title = r['title']
            filtered.append(title)    
    
    return filtered

def tokenize(text):
    return set(re.findall(r"\w+", text.lower()))

def normalize_text(text):
    """Normalize text for comparison."""
    return " ".join(text.lower().split())

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
    
    results = send_sparql_request(query_1)
    res_output = [r for r in results if 'entity/Q' in r['item']['value']]
    if res_output:
        return res_output

    results = send_sparql_request(query_2)
    res_output = [r for r in results if 'entity/Q' in r['item']['value']]
    if res_output:
        return res_output

    print("No results found after both searches.")
    return []
    
def extract_qid(url):
    return url.split('/')[-1]

def send_sparql_request(query, max_retries=10):
    endpoint_url = "https://query.wikidata.org/sparql"
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "MyWikidataSearchApp/1.0"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(endpoint_url, params={"query": query}, headers=headers)
            
            # nếu thành công
            if response.status_code == 200:
                return response.json()["results"]["bindings"]
            
            # nếu bị rate-limit (429)
            elif response.status_code == 429:
                wait_time = 2 ** attempt
                print(f"⚠️ Rate limit (429). Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            
            # nếu là lỗi tạm thời khác (5xx)
            elif 500 <= response.status_code < 600:
                wait_time = 2 ** attempt
                print(f"⚠️ Server error {response.status_code}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            else:
                print(f"❌ Query failed with status {response.status_code}: {response.text}")
                return []

        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}. Retrying...")
            time.sleep(2 ** attempt)
            continue

    print("❌ Max retries reached. Returning empty result.")
    return []

def get_relation_id(q_relation, q_relation_child=None, type_q=None):
    """
    Args:
        q_relation: The main relation keyword to search for
        q_relation_child: Optional child relation keyword for bidirectional queries
        type_q: Type of question ("inference" or "compositional")
                - "inference": Uses dict_p_relation (bidirectional properties only)
                - "compositional": Uses PROPERTY_MAPPINGS, but switches to bidirectional
                  if the property matches dict_p_relation (P22, P25, P26, P40, P106, P3373)
    
    Returns:
        - For bidirectional properties:
          (all_relation_string, [parent_property_id], [child_property_ids])
        - For unidirectional properties:
          (None, [property_id], [])
        - For unknown properties:
          (None, [], [])
    """
    search_normalized = normalize_text(q_relation)
    
    # For inference questions: use dict_p_relation (bidirectional)
    if type_q == "inference":
        # Find parent relation keyword from dict_p_relation
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
            return None, [], []

        p_relation = dict_p_relation.get(q_relation)
        print("p_relation: ", p_relation)
        if not p_relation:
            return None, [], []
        
        # Find child relation if provided
        c_relation = []
        if q_relation_child:
            if "father" in q_relation_child.lower() or "mother" in q_relation_child.lower():
                c_relation = ["child"]
            elif "spouse" in q_relation_child.lower() or "wife" in q_relation_child.lower() or "husband" in q_relation_child.lower():
                c_relation = ["spouse"]
            elif "child" in q_relation_child.lower() or "son" in q_relation_child.lower() or "daughter" in q_relation_child.lower():
                c_relation = ["mother", "father"]
            elif "sibling" in q_relation_child.lower() or "brother" in q_relation_child.lower() or "sister" in q_relation_child.lower():
                c_relation = ["sibling"]
            elif "work" in q_relation_child.lower():
                c_relation = ["occupation"]

        # Build SPARQL relation string
        all_relation = "wd:" + p_relation
        cid_relation = []
        if c_relation:
            for r in c_relation:
                if r == q_relation:
                    continue
                cid = dict_p_relation.get(r.lower())
                if cid:
                    all_relation += " wd:" + cid
                    cid_relation.append(cid)
        
        return all_relation, [p_relation], cid_relation
    
    # For compositional questions: use PROPERTY_MAPPINGS (unidirectional)
    elif type_q == "compositional":
        # Direct match in PROPERTY_MAPPINGS
        property_id = None
        if search_normalized in PROPERTY_MAPPINGS:
            property_id = PROPERTY_MAPPINGS[search_normalized]
        else:
            # Partial match in PROPERTY_MAPPINGS
            for keyword, prop_id in PROPERTY_MAPPINGS.items():
                if keyword in search_normalized or search_normalized in keyword:
                    property_id = prop_id
                    break
        
        # If no property found, return empty
        if not property_id:
            return None, [], []
        
        # Check if this property is a bidirectional property (in dict_p_relation)
        # If yes, use bidirectional mode for compositional as well
        if property_id in BIDIRECTIONAL_PROPERTIES:
            # This is a bidirectional property, treat it like inference
            # Normalize q_relation to match dict_p_relation keys
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
            elif "work" in q_relation.lower() or "occupation" in q_relation.lower():
                q_relation = "occupation"
            else:
                # Property is bidirectional but relation not recognized, use unidirectional
                return None, [property_id], []
            
            p_relation = dict_p_relation.get(q_relation)
            if not p_relation:
                return None, [property_id], []
            
            # Find child relation if provided
            c_relation = []
            if q_relation_child:
                if "father" in q_relation_child.lower() or "mother" in q_relation_child.lower():
                    c_relation = ["child"]
                elif "spouse" in q_relation_child.lower() or "wife" in q_relation_child.lower() or "husband" in q_relation_child.lower():
                    c_relation = ["spouse"]
                elif "child" in q_relation_child.lower() or "son" in q_relation_child.lower() or "daughter" in q_relation_child.lower():
                    c_relation = ["mother", "father"]
                elif "sibling" in q_relation_child.lower() or "brother" in q_relation_child.lower() or "sister" in q_relation_child.lower():
                    c_relation = ["sibling"]
                elif "work" in q_relation_child.lower():
                    c_relation = ["occupation"]
            
            # Build SPARQL relation string (bidirectional mode)
            all_relation = "wd:" + p_relation
            cid_relation = []
            if c_relation:
                for r in c_relation:
                    if r == q_relation:
                        continue
                    cid = dict_p_relation.get(r.lower())
                    if cid:
                        all_relation += " wd:" + cid
                        cid_relation.append(cid)
            
            return all_relation, [p_relation], cid_relation
        
        # Unidirectional mode - just return the property ID
        return None, [property_id], []
    
    # Unknown type_q
    else:
        return None, [], []

def sparql_find_parent(qid, all_relation):
    """
    Unified SPARQL query function for both bidirectional and unidirectional property queries.
    
    This single query format works for both single and multiple property IDs.
    
    Args:
        qid: Wikidata entity ID (e.g., "Q272899")
        all_relation: Property ID(s) with "wd:" prefix:
            - Single property: "wd:P19"
            - Multiple properties: "wd:P22 wd:P25" (space-separated)
    
    Returns:
        List of SPARQL query results with ?Pid, ?PidLabel, ?property, ?propertyLabel
    """
    # Ensure all_relation has wd: prefix
    if not all_relation.startswith('wd:'):
        all_relation = 'wd:' + all_relation
    
    query = f"""
    SELECT ?Pid ?PidLabel ?property ?propertyLabel
    WHERE {{
        wd:{qid} ?propIRI ?property .
        ?Pid wikibase:directClaim ?propIRI .
        VALUES ?Pid {{ { all_relation } }}
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    
    return send_sparql_request(query)

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

def query_hop2_with_wikidata(type_q, original_question, q_key_entity, last_key_entity, q_relation, c_relation, 
                             retrieval_inf, last_retrieval_inf):
    """
    Integrated query function supporting both bidirectional and unidirectional property queries.
    
    Query mode is determined by question type and property characteristics:
    - "inference": Always uses bidirectional mode (dict_p_relation only)
    - "compositional": Intelligently selects mode based on property:
      * Bidirectional if property matches dict_p_relation (father, mother, spouse, child, sibling, occupation)
      * Unidirectional otherwise (uses PROPERTY_MAPPINGS for other properties)
    
    Args:
        type_q: Type of question ("inference" or "compositional")
                - "inference": Forces bidirectional mode with verification
                - "compositional": Auto-selects mode based on property type
        original_question: The original question text (for special case handling)
        q_key_entity: The main entity to query (hop 1)
        last_key_entity: The related entity for verification (hop 0, required for bidirectional)
        q_relation: The relationship/property keyword for hop 1
        c_relation: The relationship keyword for hop 0 (for bidirectional verification)
        retrieval_inf: Retrieval information for hop 1
        last_retrieval_inf: Retrieval information for hop 0 (for bidirectional verification)
    
    Returns:
        dict_output with answer, source, and verification information
    """
    dict_output = {
        "id": {},
        "hop_0": {},
        "hop_1": {},
        "parent": {}, 
        "verify": {},
        "is_bidirectional": False
    }
    
    # Get relation/property information using unified function
    # Pass type_q to get_relation_id so it knows which mapping to use
    all_relation, pid_relation, cid_relation = get_relation_id(q_relation, c_relation, type_q=type_q)
    
    # Determine if this should be bidirectional or unidirectional
    # - "inference": always bidirectional (uses dict_p_relation)
    # - "compositional": can be bidirectional OR unidirectional
    #   * bidirectional if property matches dict_p_relation (all_relation is not None)
    #   * unidirectional otherwise (all_relation is None)
    if type_q == "inference":
        # For inference, all_relation must be present (bidirectional mode)
        if all_relation is None:
            # Property not recognized as bidirectional - return empty for inference
            dict_output["is_bidirectional"] = False
            dict_output['long_answer'] = ''
            dict_output['short_answer'] = ''
            dict_output["verify"] = []
            return dict_output
        is_bidirectional = True
    elif type_q == "compositional":
        # For compositional, check if all_relation is present
        # If present, it's a bidirectional property (matches dict_p_relation)
        # If None, it's a unidirectional property
        is_bidirectional = all_relation is not None
    else:
        # Unknown type_q
        dict_output["is_bidirectional"] = False
        dict_output['long_answer'] = ''
        dict_output['short_answer'] = ''
        dict_output["verify"] = []
        return dict_output
    
    dict_output["is_bidirectional"] = is_bidirectional
    
    # If no valid property found, return empty result with required keys
    if not pid_relation:
        dict_output['long_answer'] = ''
        dict_output['short_answer'] = ''
        dict_output["verify"] = []
        return dict_output
    
    dict_output["id"]["pid"] = pid_relation
    dict_output["id"]["cid"] = cid_relation
    
    # Normalize and search for hop 1 entity (main entity)
    key_entity = norm_key_entity(q_key_entity, retrieval_inf)
    print("norm key_entity: ", key_entity)
    
    list_qid = search_sparql(key_entity)

    if len(list_qid) == 0:
        q_search_wikidata = search_google(key_entity)
        if q_search_wikidata and len(q_search_wikidata) == 0:
            q_search_wikidata = search_google(q_key_entity)
        
        if q_search_wikidata:
            list_norm_title_parent = norm_keyentity_wikidata_api(q_search_wikidata)
            key_entity = list_norm_title_parent[0] if list_norm_title_parent else key_entity
            for title in list_norm_title_parent:
                list_qid.extend(search_sparql(title))
    
    # Normalize and search for hop 0 entity (for verification, if bidirectional)
    norm_last_key_entity = None
    list_qid_last_key = []
    
    if is_bidirectional:
        # For bidirectional mode, we must have last_key_entity
        if not last_key_entity:
            print("Warning: Bidirectional mode requires last_key_entity, but it's missing")
        else:
            norm_last_key_entity = norm_key_entity(last_key_entity, last_retrieval_inf)
            list_qid_last_key = search_sparql(norm_last_key_entity)

            if len(list_qid_last_key) == 0:
                q_search_wikidata = search_google(norm_last_key_entity)
                if q_search_wikidata and len(q_search_wikidata) == 0:
                    q_search_wikidata = search_google(last_key_entity)

                if q_search_wikidata:
                    list_norm_title_parent = norm_keyentity_wikidata_api(q_search_wikidata)
                    norm_last_key_entity = list_norm_title_parent[0] if list_norm_title_parent else last_key_entity
                    for title in list_norm_title_parent:
                        list_qid_last_key.extend(search_sparql(title))
    
    dict_output["hop_0"] = {
        "key_entity": norm_last_key_entity,
        "list_qid": list_qid_last_key,
    }
    dict_output["hop_1"] = {
        "key_entity": key_entity,
        "list_qid": list_qid,
    }
    
    # Query properties for each candidate QID
    check_qid_parent = set()
    for item in list_qid:
        qid = extract_qid(item["item"]["value"])
        if not re.match(r"^Q\d+$", qid):
            print(f"'{qid}' is a wikipedia link")
            continue
        
        if qid in check_qid_parent:
            continue
        check_qid_parent.add(qid)
        
        # For bidirectional: use the all_relation approach
        if is_bidirectional and all_relation:
            parent = sparql_find_parent(qid, all_relation)
            dict_output["parent"][qid] = parent
        # For unidirectional: query the specific property directly
        elif not is_bidirectional and pid_relation:
            # Get the property ID from pid_relation and add wd: prefix
            property_id = pid_relation[0]
            # Query the property values directly using sparql_find_parent
            results = sparql_find_parent(qid, f"wd:{property_id}")
            
            # Extract label values from SPARQL results (using propertyLabel, not valueLabel)
            property_values = []
            for result in results:
                if "propertyLabel" in result:
                    label = result["propertyLabel"]["value"]
                    if label and not label.startswith("Q"):  # Exclude QID itself
                        property_values.append(label)
            
            # Handle special cases for questions
            if original_question:
                if 'work at' in original_question.lower():
                    # Use get_relation_id to find employer property
                    _, employer_ids, _ = get_relation_id('employer', type_q='compositional')
                    if employer_ids:
                        results = sparql_find_parent(qid, f"wd:{employer_ids[0]}")
                        property_values = []
                        for result in results:
                            if "propertyLabel" in result:
                                label = result["propertyLabel"]["value"]
                                if label and not label.startswith("Q"):
                                    property_values.append(label)
                elif 'nationality' in original_question.lower():
                    # Try both nationality and country of origin
                    if not property_values:
                        _, alt_property_ids, _ = get_relation_id('country of origin', type_q='compositional')
                        if alt_property_ids:
                            results = sparql_find_parent(qid, f"wd:{alt_property_ids[0]}")
                            property_values = []
                            for result in results:
                                if "propertyLabel" in result:
                                    label = result["propertyLabel"]["value"]
                                    if label and not label.startswith("Q"):
                                        property_values.append(label)
            
            # If we found values, match with retrieval info or take first value
            if property_values:
                short_answer = ''
                
                # Match with retrieval_inf if provided
                if retrieval_inf:
                    for val in property_values:
                        if val and val.lower() in retrieval_inf.lower():
                            short_answer = val
                            break
                
                # If no match or no retrieval_inf, take first value
                if not short_answer:
                    short_answer = property_values[0]
                
                # Store result and return immediately for unidirectional
                dict_output["parent"][qid] = [
                    {
                        "Pid": {"value": f"http://www.wikidata.org/entity/{property_id}"},
                        "property": {"value": f"http://www.wikidata.org/entity/{qid}"},
                        "propertyLabel": {"value": short_answer}
                    }
                ]
                dict_output["verify"] = [{
                    "qid": qid,
                    "parent_relations": {
                        property_id: {
                            "qid": qid,
                            "value": short_answer
                        }
                    },
                    "child_relations": {}
                }]
                dict_output['long_answer'] = ''
                dict_output['short_answer'] = short_answer
                
                # For unidirectional, return immediately without verify_hop2
                return dict_output
    
    # For bidirectional queries, verify and extract answer
    if is_bidirectional:
        return verify_hop2(dict_output)
    
    # If unidirectional but no answer found, return empty result
    dict_output['long_answer'] = ''
    dict_output['short_answer'] = ''
    dict_output["verify"] = []
    return dict_output

def verify_hop2(dict_output):
    verify_list = []
    parent_id = dict_output["id"]["pid"]
    child_id = dict_output["id"]["cid"]
    child_info = dict_output['hop_0']
    short_answer  = ''
    # duyệt qua từng parent và danh sách con của nó
    for parent_qid, sample in dict_output.get("parent", {}).items():
        p_value = {}
        c_value = {}
        # ok = 0
        for item in sample:
            pid = item["Pid"]["value"].split("/")[-1] # lấy Pid
            qid = item["property"]["value"].split("/")[-1] # lấy QID của child
            propertyLabel = item["propertyLabel"]["value"] if "propertyLabel" in item else None
            ok = 0
            if pid in parent_id:
                p_value[pid] = {
                    "qid": qid,
                    "value": propertyLabel
                }
                tmp_ans = propertyLabel
                for item_c in sample:
                    pid = item_c["Pid"]["value"].split("/")[-1] # lấy Pid
                    qid = item_c["property"]["value"].split("/")[-1] # lấy QID của child
                    propertyLabel = item_c["propertyLabel"]["value"] if "propertyLabel" in item_c else None
                    
                    if pid in child_id:                
                        if propertyLabel.lower() == child_info['key_entity'].lower():
                            ok = 1
                            short_answer = tmp_ans
                            c_value[pid] = {
                                "qid": qid,
                                "value": propertyLabel
                            }
                            break
                    
            if ok == 1:
                verify_list.append({
                    "qid": parent_qid,
                    "parent_relations": p_value,
                    "child_relations": c_value
                })   
                break

    # thêm vào dict_output
    dict_output["verify"] = verify_list
    dict_output['long_answer'] = ''
    dict_output['short_answer'] = short_answer
    
    return dict_output

if __name__ == "__main__":
    # Test bidirectional mode (multiple properties)
    print("Bidirectional query:")
    print(sparql_find_parent("Q272899", "wd:P22 wd:P25"))
    
    # Test unidirectional mode (single property)
    print("\nUnidirectional query:")
    print(sparql_find_parent("Q272899", "wd:P19"))