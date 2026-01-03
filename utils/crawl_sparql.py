import re
from SPARQLWrapper import JSON, SPARQLWrapper
import requests
import time
import sys
import os
from bs4 import BeautifulSoup
from utils.util import caculate_scores
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/110.0.0.0 Safari/537.36"
    )
}

def get_last_oldid_before_date(title: str, date: str = "2021-04-07"):
    """
    Lấy oldid cuối cùng trước ngày date (YYYY-MM-DD) của entity (title = Qxxx).
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,
        "rvprop": "ids|timestamp",
        "rvdir": "older",   # lấy bản cũ hơn so với rvstart
        "rvstart": date + "T00:00:00Z",
        "format": "json"
    }
    headers = {
        "User-Agent": "MinhThu/1.0 (https://22028116@vnu.edu.vn)"
    }

    res = requests.get(url, params=params, headers=headers)
    res.raise_for_status()
    data = res.json()

    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        revisions = page.get("revisions", [])
        if revisions:
            rev = revisions[0]
            return rev["revid"]

    return None

def get_properties_values(qid, dict_pid_date):
    oldid = get_last_oldid_before_date(qid)
    if oldid:
        print(f"Oldid cuối cùng: {oldid}")
        url = f"https://www.wikidata.org/w/index.php?title={qid}&oldid={oldid}"
    else:
        url = f"https://www.wikidata.org/wiki/{qid}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    
    for pid, item in dict_pid_date.items():
        group = soup.find("div", {"id": pid})
        values = []
        if not group:
            item['norm_values'] = values
            continue
        
        mainsnaks = group.find_all("div", class_="wikibase-statementview-mainsnak-container")
        for snak in mainsnaks:
            val_div = snak.find("div", class_="wikibase-snakview-value")
            if val_div:
                values.append(val_div.get_text(strip=True))
        item['norm_values'] = values

    return dict_pid_date


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

def normalize_text(text):
    text = re.sub(r'\s*\([^)]*\)\s*', '', text)
    return text.strip()

def norm_key_entity_search(entity_name, cos_threshold=0.5, fuzz_threshold=0.5):
    """
    Chuẩn hóa tên entity bằng cách tìm trên Wikidata, tính điểm cosine/fuzzy và chọn entity tốt nhất.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity_name,
        "language": "en",
        "format": "json",
        "limit": 10
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, params=params, headers=headers, timeout=60)
    data_entity = response.json().get("search", [])
    if not data_entity:
        print(f"Không tìm thấy entity nào cho '{entity_name}'.")
        return {"error": f"Không tìm thấy entity nào cho '{entity_name}'."}
    
    # 2. Lấy label của từng QID (tên hiển thị)
    headers = {"User-Agent": "Mozilla/5.0"}
    entities = []
    for item in data_entity:
        qid = item["id"]
        label = item["label"]
        entities.append(label)
    

    if not entities:
        return {"error": f"Không lấy được label của bất kỳ entity nào cho '{entity_name}'."}

    # 3. Tính điểm cosine/fuzzy và chọn entity tốt nhất
    scored_entities = []
    for ent in entities:
        scores = caculate_scores(entity_name, ent)
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
        return {"error": f"Không tìm thấy entity nào cho '{entity_name}'."}
    
    # print(f'Các entity: {entities} -> Chọn "{best["entity"]}"')
    return best["entity"]

def extract_qid(url):
    return url.split("/")[-1]

def remove_parentheses(text):
    # cleaned = re.sub(r"\s*\([^)]*\)", "", text).strip()
    # return cleaned
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    
    # Kiểm tra nếu string rỗng
    if not text.strip():
        return ""
    # 1. Xóa nội dung trong dấu ngoặc đơn () VÀ chính dấu ngoặc đơn đó.
    #    Regex: r'\s*\([^)]*\)' tìm kiếm () và nội dung bên trong, bao gồm cả khoảng trắng trước đó.
    cleaned_parentheses = re.sub(r'\s*\([^)]*\)', '', text)
    
    # 2. Xóa các ký tự đặc biệt khác: ", \, *, / (Lưu ý: giữ lại dấu phẩy ,)
    #    Lớp ký tự: [\"\\*/]
    #    Thay thế các ký tự này bằng MỘT KHOẢNG TRẮNG để tránh gộp từ.
    cleaned_chars = re.sub(r'["\\*/]', ' ', cleaned_parentheses)
    
    # 3. Xử lý trường hợp chỉ có dấu mở ngoặc đơn '(' (và các ký tự sau đó)
    #    Tìm dấu '(' và xóa toàn bộ phần còn lại của chuỗi.
    if '(' in cleaned_chars:
        cleaned_chars = cleaned_chars.split('(')[0]
    
    # 4. Chuẩn hóa khoảng trắng: thay thế nhiều khoảng trắng thành một và xóa khoảng trắng ở đầu/cuối.
    #    Bước này khắc phục vấn đề "EricViking" bị gộp.
    final_cleaned = re.sub(r'\s+', ' ', cleaned_chars).strip()
    
    return final_cleaned

def fill_all_qid_search_sparql(keyword, limit=20, retries=5, delay=5, language="en",):
    safe_keyword = keyword.replace("\\", "\\\\").replace('"', '\\"').strip()
    if not safe_keyword:
        return [], ""

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    query = f"""
    SELECT ?item ?itemLabel ?itemDescription WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:endpoint "www.wikidata.org";
                        wikibase:api "EntitySearch";
                        mwapi:search "{safe_keyword}";
                        mwapi:language "{language}".
        ?item wikibase:apiOutputItem mwapi:item.
      }}
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "{language}".
      }}
    }}
    LIMIT {limit}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    for attempt in range(1, retries + 1):
        try:
            results = sparql.query().convert()
            break
        except Exception as e:
            print(f"[WARN] Attempt {attempt}/{retries} failed for keyword: {keyword}\n→ {e}")
            if attempt < retries:
                time.sleep(delay * attempt)
            else:
                return [], ""
    else:
        return [], ""

    entities = []
    qid_description = ""
    for result in results.get("results", {}).get("bindings", []):
        entity_id = result["item"]["value"].split("/")[-1]
        label = result.get("itemLabel", {}).get("value", "")
        description = result.get("itemDescription", {}).get("value", "")
        entities.append({
            "qid": entity_id,
            "label": label,
            "description": description
        })
        if description != '':
            qid_description += f"- {description}\n"

    time.sleep(delay)

    return entities, qid_description

def find_all_qid(key_entity, limit=20):  # add description
    key_entity = remove_parentheses(key_entity)
    
    search_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": key_entity,
        "language": "en",
        "format": "json",
        "limit": limit  
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
        url = item.get("concepturi", "")  

        qid_info.append({
            "qid": qid,
            "label": label,
            "description": description,
            "url": url
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
    key_entity = norm_key_entity_search(entity)
    if not key_entity or isinstance(key_entity, dict) and "error" in key_entity:
        print(f"❌ Không chuẩn hóa được entity: {entity}")
        key_entity = entity

    qids = find_all_qid(key_entity)
    if not qids:
        return ""

    qid_value = best_match_qid(qids, key_entity)
    best_qid = qid_value
        
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
    # print(result)
    dict_pid_date = {}
    for item in result:
        property_label = item.get("propertyLabel", {}).get("value", "")
        value_label = item.get("valueLabel", {}).get("value", "")
        value_description = item.get("valueDescription", {}).get("value", "")
        # Lấy pid từ sparql, lấy value từ html
        # if 'date' in property_label.lower():
        pid_property = item.get("property", {}).get("value", "").split("/")[-1]
        dict_pid_date[pid_property] = {
            'property_label': property_label,
            'value_label': value_label,
            'value_description': value_description
        }
        
        # else:
        #     if value_description == "":
        #         triple = entity + " " + property_label + " " + value_label
        #     else:
        #         triple = entity + " " + property_label + " " + value_label + ", description: " + value_description
        #     sparql_context.append(triple)
    
    #norm values with wikidata api
    if dict_pid_date:
        # pids = list(dict_pid_date.keys())
        dict_pid_date = get_properties_values(best_qid, dict_pid_date)

        for pid, item in dict_pid_date.items():
            if pid in dict_pid_date:
                norm_value = item['norm_values'][0] if item['norm_values'] else ""
                if norm_value:
                    property_label = item['property_label']
                    value_description = item['value_description']
                    if value_description == "":
                        triple = entity + " " + property_label + " " + norm_value
                    else:
                        triple = entity + " " + property_label + " " + norm_value + ", description: " + value_description
                    sparql_context.append(triple)
                    
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
    
    return key_entity + ": " + "\n".join(selected)

if __name__ == "__main__":
    # id = "ef32ebf40bdc11eba7f7acde48001122"
    # entity = "Archduke Hubert Salvator of Austria" #"Princess Rosemary of Salm-Salm" #"Emich, 5th Prince of Leiningen"
    # result = find_sparql_context(entity)
    
    # # ranked_docs = find_most_relevant_doc_embedding("husband of Princess Rosemary of Salm-Salm", result)
    # # ranked_docs = find_most_relevant_doc("spouse, husband of Princess Rosemary of Salm-Salm", result)
    
    # ranked_docs = find_most_relevant_doc("place of burial", result)
    # print(select_top_documents(entity, ranked_docs))
    q_relation = "father of Salm-Salm"
    entity_names = [
    'Eric Longlegs, Lord of Langeland',
    '\"Greatest Love',
    'Eric (The Great)/Viking',
    'Item with (missing close'
]
    for item in entity_names:
        print(remove_parentheses(item))
    