from src.gpt_request import request_api
from src.prompt import load_prompt_gen_sub
from src.split_question import split_comparison_question
from utils.utils import load_json, save_json
import json
import os 

def generate_subquestion(question, sample_id, filepath, is_seq=True, dataset=None, type_q=None, model="gpt-4o-mini"):
    if os.path.exists(filepath):
        output = load_json(filepath)
        return output
    
    messages = load_prompt_gen_sub(question, is_seq=is_seq, dataset=dataset, type_q=type_q)
    result_json = request_api(messages, model)
    
    if not result_json or result_json.strip() == "":
        raise ValueError("Empty response from API")
        
    start_idx = result_json.find("[")
    end_idx = result_json.rfind("]") + 1
    result_json = result_json[start_idx:end_idx]
        
    try:
        search_results = json.loads(result_json)
        output = {}
        for step, sub in enumerate(search_results):
            if "key_entity" in sub:
                key_entity = sub["key_entity"]
            else:
                key_entity = None
            
            if "subquestion" in sub:
                subquestion = sub["subquestion"]
            else:
                subquestion = question # non-seq
                
            output[step] = {
                "QIDX": sample_id,
                "step_number": step,
                "subquestion": subquestion,
                "search_keyword": sub['search_keyword'],
                "key_entity": key_entity
            }
                           
        save_json(output, filepath)
        return output
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response in generate_subquestion: {e}")
        print(f"Raw response: {result_json}")
        raise ValueError(f"Invalid JSON response from API: {result_json}")
    
def question_decomposition(question, dataset=None, type_q=None):
    if dataset != 'wiki':
        return [question]
    
    if type_q in ['comparison', 'bridge_comparison']:
        return split_comparison_question(question)
    else:
        return [question]
        
    