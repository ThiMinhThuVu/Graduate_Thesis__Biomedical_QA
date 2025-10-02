from src.prompt import *
from src.gpt_request import *
from src.retrieval import norm_entity_with_keycontext
from utils.util import load_json
import json

def generate_answer(question, key_entities, relevant_info, dataset=None, type_q=None, model="o3-mini"):    
    prompt = load_prompt_gen_ans(question, key_entities, relevant_info, dataset=dataset, type_q=type_q)
    if dataset in ["cwq", "wiki"]:
        messages = [
            {"role": "system", "content": "You are an expert in general knowledge. Your task is to provide accurate answers based on the provided information."},
            {"role": "user", "content": prompt}
        ]
    else:
        messages=[
            {"role": "system", "content": "You are an expert in medical and scientific knowledge. Your task is to provide accurate answers based on the provided information."},
            {"role": "user", "content": prompt}
        ]
    
    result_json = request_api(messages, model)
    
    try:
        result_dict = json.loads(result_json)
        # long_answer = result_dict.get("long_answer", "")
        # short_answer = result_dict.get("short_answer", "")
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {result_json}")

        long_answer = ""
        short_answer = ""
        
        if "long_answer" in result_json.lower() and "short_answer" in result_json.lower():
            parts = result_json.split("long_answer")
            if len(parts) > 1:
                remaining = parts[1].strip()
                if "short_answer" in remaining:
                    answers = remaining.split("short_answer")
                    long_answer = answers[0].strip().strip('":').strip()
                    short_answer = answers[1].strip().strip('":').strip()
        result_dict = {
            "long_answer": long_answer,
            "short_answer": short_answer
        }
 
    # return {
    #     "long_answer": long_answer,
    #     "short_answer": short_answer
    # }
    return result_dict

def generate_combine_parallel_answers(question, context, dataset='wiki', model="o3-mini"):
    prompt = prompt_gen_combine_parallel_answers(question, context)
        
    messages = [
        {"role": "system", "content": "You are an expert in general knowledge. Your task is to provide accurate answers based on the provided information. Please only output the choice (without additional explanations)"},
        {"role": "user", "content": prompt}
    ]
    
    result_json = request_api(messages, model)
    
    try:
        result_dict = json.loads(result_json)
        long_answer = result_dict.get("long_answer", "")
        short_answer = result_dict.get("short_answer", "")
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {result_json}")

        long_answer = ""
        short_answer = ""
        
        if "long_answer" in result_json.lower() and "short_answer" in result_json.lower():
            parts = result_json.split("long_answer")
            if len(parts) > 1:
                remaining = parts[1].strip()
                if "short_answer" in remaining:
                    answers = remaining.split("short_answer")
                    long_answer = answers[0].strip().strip('":').strip()
                    short_answer = answers[1].strip().strip('":').strip()
        
        if not long_answer:
            long_answer = result_json
            short_answer = "No relevant answer found"
    
    return {
        "long_answer": long_answer,
        "short_answer": short_answer
    }

def gen_ans_multiple_choice(question, context, dataset='medhop', model = "o3-mini"):
    prompt = f"""Question: {question}
        Instructions:
        - Please base your answer ONLY on the context provided.
        - Please only choose the given answers (without additional explanations). For example: "A", "B", "C", "D", "E", "F", "G".
        - Do not include anything else aside from your answer.

        Context:
        {context}
        Please provide your short answer to the question based on the context above."""
    if dataset in ['cwq', 'wiki']:
        messages = [
            {"role": "system", "content": "You are an expert in general knowledge. Your task is to provide accurate answers based on the provided information. Please only output the choice (without additional explanations)"},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "system", "content": "You are an expert in medical and scientific knowledge. Your task is to provide accurate answers based on the provided information. Please only output the choice (without additional explanations)"},
            {"role": "user", "content": prompt}
        ]
    
    result_json = request_api(messages, model)
    
    return result_json

def gen_evidences(sample, data, type_q, model="gpt-4o-mini"):    
    q_ids = sorted([k for k in data.keys() if k.isdigit()], key=int)
    list_subquestion = []
    dict_evidences = {}
    cnt_evd = 0
    
    for q_id in q_ids:
        q_id = str(q_id)
        subdata = data[q_id]
        last_path = str(len(subdata.keys()) - 2)
        datapath = subdata[str(last_path)]
        last_step = str(len(datapath.keys()) - 1)
        
        for step, subitem in datapath.items():
            e1 = subitem['key_entity']
            e2 = subitem['short_answer']
            list_subquestion.append(subitem['question'])
            e1 = norm_entity_with_keycontext(e1, sample)
            if step != last_step:
                e2 = norm_entity_with_keycontext(e2, sample)
            
            dict_evidences[str(cnt_evd)] = {
                "e1": e1,
                "e2": e2
            }
            cnt_evd += 1

    messages = prompt_gen_relation(list_subquestion, type_q)
    result_json = request_api(messages, model)
    
    start_idx = result_json.find("[")
    end_idx = result_json.rfind("]") + 1
    result_json = result_json[start_idx:end_idx]
    
    try:
        relation_predict = json.loads(result_json)
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {result_json}")
    
    list_evidences = []
    for idx, item in dict_evidences.items():
        rel = relation_predict[int(idx)]
        list_evidences.append([item['e1'], rel, item['e2']])

    return list_evidences
