from src.prompt import *
from src.gpt_request import *
import json

def generate_answer(question, key_entities, relevant_info, dataset="medhop", model="o3-mini"):    
    prompt = load_prompt_gen_ans(question, key_entities, relevant_info, dataset=dataset)
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

def gen_ans_compare_wiki(question, context, dataset='wiki', model = "o3-mini"):
    prompt = prompt_gen_answer_wiki_step_3b_comparison(question, context)
        
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

