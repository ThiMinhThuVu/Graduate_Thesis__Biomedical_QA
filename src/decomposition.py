from src.gpt_request import request_api
from src.prompt import load_prompt_gen_sub, prompt_split_question
from src.split_question import split_comparison_question, split_bridge_comparison_fallback
from utils.util import load_json, save_json
import json
import os 

RELATION_PATHS = {
    # Quan hệ thông qua hôn nhân (In-laws and extended)
    "co-husband": [
        ("spouse", "spouse")
    ],
    "co-wife": [
        ("spouse", "spouse")
    ],
    "father-in-law": [
        ("spouse", "father")
    ],
    "mother-in-law": [
        ("spouse", "mother")
    ],
    "sibling-in-law": [
        ("spouse", "sibling"),
        ("sibling", "spouse")
    ],
    "child-in-law": [
        ("child", "spouse")
    ],
    "stepchild": [
        ("spouse", "child")
    ],
    "stepmother": [
        ("father", "spouse")
    ],
    "stepfather": [
        ("mother", "spouse")
    ],

    # Quan hệ huyết thống (Blood relatives)
    "paternal grandfather": [
        ("father", "father")
    ],
    "paternal grandmother": [
        ("father", "mother")
    ],
    "maternal grandfather": [
        ("mother", "father")
    ],
    "maternal grandmother": [
        ("mother", "mother")
    ],
    "grandchild": [
        ("child", "child")
    ],
    "uncle": [
        ("father", "sibling"),
        ("mother", "sibling")
    ],
    "aunt": [
        ("father", "sibling"),
        ("mother", "sibling")
    ],
    "father": [
        ("mother", "spouse"),
        ("sibling", "father")
    ],
    "mother": [
        ("father", "spouse"),
        ("sibling", "mother")
    ],
    "sibling": [
        ("father", "child"),
        ("mother", "child"),
        ("sibling", "sibling")
    ],
    "husband": [
        ("child", "father")
    ],
    "wife": [
        ("child", "mother")
    ],
     "child": [
        ("spouse", "child"),
        ("child", "sibling")
    ],
     #Ngoại các rule khác
    "married to": [
        ("child", "father"),
        ("child", "mother")
    ],
    "married": [
        ("child", "father"),
        ("child", "mother")
    ],

    # Quan hệ học thuật/công việc (Academic/Work)
    "employer": [
        ("doctoral student", "educated at")
    ],
    "educated at": [
        ("doctoral advisor", "employer")
    ],
    "field of work": [
        ("doctoral advisor", "field of work"),
        ("doctoral student", "field of work")
    ],
    "field of study": [
        ("doctoral advisor", "field of study"),
        ("doctoral student", "field of study")
    ]
}

def create_relation_path_inference(entity, relation, sample_id, target_label):
    output = {}
    n_path = 0
    if relation in RELATION_PATHS:
        R = RELATION_PATHS[relation]
        n_path = len(R)
        for idx, r in enumerate(R):            #idx là way0, way 1
            sub0 = f"Who is the {r[0]} of {entity}?"
            if relation in ['employer', 'educated at', 'field of work']:
                sub1 = f"What is the {r[1]} of that person?"
                target_label = ['ORG']
            else:
                sub1 = f"Who is the {r[1]} of that person?"
                target_label = ['PERSON'] 
                   
            s0 = f"{r[0]} of {entity}"
            s1 = f"{r[1]} of"
            
            output[str(idx)] = { #way idx
                "0": {
                    "QIDX": sample_id,
                    "step_number": 0,
                    "subquestion": sub0,
                    "search_keyword": s0,
                    "key_entity": [entity],
                    "target_labels": target_label
                }, 
                "1": {
                    "QIDX": sample_id,
                    "step_number": 1,
                    "subquestion": sub1,
                    "search_keyword": s1,
                    "key_entity": [],
                    "target_labels": target_label
                }
            }
    else:
        raise ValueError(f"Unknown relation: {relation}")
    
    return output, n_path
    
def generate_subquestion(question, sample_id, filepath, is_seq=True, dataset=None, type_q=None, model="gpt-4o-mini"):
    if os.path.exists(filepath):
        output = load_json(filepath)
        n_path = len(output.keys())
        return output, n_path
    
    messages = load_prompt_gen_sub(question, is_seq=is_seq, dataset=dataset, type_q=type_q)
    result_json = request_api(messages, model)
    
    if not result_json or result_json.strip() == "":
        raise ValueError("Empty response from API")
        
    start_idx = result_json.find("[")
    end_idx = result_json.rfind("]") + 1
    result_json = result_json[start_idx:end_idx]
            
    try:
        search_results = json.loads(result_json)
        
        if type_q == "inference":
            key_entity = search_results[0]['key_entity']
            relation_entity = search_results[0]['key_relation']
            target_label = search_results[0]['target_labels']
            output, n_path = create_relation_path_inference(key_entity, relation_entity, sample_id, target_label)
        else:
            n_path = 1
            output = {"0": {}}
            
            for step, sub in enumerate(search_results):
                if "key_entity" in sub and sub['key_entity'] != "":
                    key_entity = [sub["key_entity"]]
                else:
                    key_entity = []
                
                if "subquestion" in sub:
                    subquestion = sub["subquestion"]
                else:
                    subquestion = question # non-seq

                output["0"][step] = {
                    "QIDX": sample_id,
                    "step_number": step,
                    "subquestion": subquestion,
                    "search_keyword": sub['search_keyword'],
                    "key_entity": key_entity,
                    "target_labels": sub["target_labels"]
                }
                
        save_json(output, filepath)
        return output, n_path
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response in generate_subquestion: {e}")
        print(f"Raw response: {result_json}")
        raise ValueError(f"Invalid JSON response from API: {result_json}")
    
def split_bridge_comparison_question(question):
    messages = prompt_split_question(question)
    response = request_api(messages, model='gpt-4o-mini')
        
    try:
        response = json.loads(response)
    except json.JSONDecodeError:
        # print("Lỗi: Không thể phân tích JSON từ phản hồi của LLM.")
        print("Dùng Regex để tách câu hỏi")
        response = split_bridge_comparison_fallback(question)
    return response

def question_decomposition(question, dataset=None, type_q=None):
    if dataset != 'wiki':
        return [question]
    
    if type_q == 'comparison':
        return split_comparison_question(question)
    elif type_q == 'bridge_comparison':
        return split_bridge_comparison_question(question)
    else:
        return [question]
        
    
if __name__ == "__main__":
    # Example usage
    question = "Do director of film Social Genocide and director of film Así No Hay Cama Que Aguante have the same nationality?"
    type_q = "bridge_comparison"

    # print(split_bridge_comparison_fallback(question))
    print(question_decomposition(question, dataset='wiki', type_q=type_q))