# from utils.process_wikidata_api import test
import json
import time
from SPARQLWrapper import JSON, SPARQLWrapper

def fill_all_qid_search_sparql(keyword, language="en", limit=20, retries=5, delay=5):
    safe_keyword = keyword.replace("\\", "\\\\").replace('"', '\\"').strip()
    if not safe_keyword:
        return []

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
                return []
    else:
        return []

    entities = []
    for result in results.get("results", {}).get("bindings", []):
        entity_id = result["item"]["value"].split("/")[-1]
        label = result.get("itemLabel", {}).get("value", "")
        description = result.get("itemDescription", {}).get("value", "")
        entities.append({
            "id": entity_id,
            "label": label,
            "description": description
        })

    time.sleep(delay)

    return entities

res = json.dumps(fill_all_qid_search_sparql("Paulo Centurión", limit=20), indent=4)
print(res)

# import nltk

# nltk.download("stopwords")
# if nltk.__version__.startswith("3.9"): 
#     nltk.download("averaged_perceptron_tagger_eng") 
# else: 
#     nltk.download("averaged_perceptron_tagger")

# from src.gpt_request import *

# def gen_ans_multiple_choice(question, context, dataset='medhop', model = "o3-mini"):
#     prompt = f"""Question: {question}
#         Instructions:
#         - Please base your answer ONLY on the context provided.
#         - Please only choose the given answers (without additional explanations). For example: "A", "B", "C", "D", "E", "F", "G", ...
#         - Do not include anything else aside from your answer.

#         Context:
#         {context}
#         Please provide your short answer to the question based on the context above."""
#     if dataset == 'wiki':
#         messages = [
#             {"role": "system", "content": "You are an expert in general knowledge. Your task is to provide accurate answers based on the provided information. Please only output the choice (without additional explanations)"},
#             {"role": "user", "content": prompt}
#         ]
#     else:
#         messages = [
#             {"role": "system", "content": "You are an expert in medical and scientific knowledge. Your task is to provide accurate answers based on the provided information. Please only output the choice (without additional explanations)"},
#             {"role": "user", "content": prompt}
#         ]
    
#     result_json = request_api(messages, model)
    
#     return result_json

# question = "When is that person's date of birth?"
# q_relation = "date of birth"
# key_entity = "Emich, 5th Prince of Leiningen"
# context = "A. Emich, 5th Prince of Leiningen (description: Prince of Leiningen (1866-1939)) - date of birth: ['18 January 1866Gregorian'] (source: 1866-01-18T00:00:00Z)\n"
# res = gen_ans_multiple_choice(question + f", {key_entity}", context, dataset='wiki', model="o3-mini")
# print(f"Answer: {res}")