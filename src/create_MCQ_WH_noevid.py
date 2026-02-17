import sys
import json
import os
import spacy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.extract_ner import extract_entities_from_evidence
from src.gpt_request import request_api
from tqdm import tqdm


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
       
def get_multiple_candidate_answer_prompt(question, search_kw, key_entities, context, max_candidates=5):
    system_content = f"""## Role
You are a biomedical expert. Your task is to generate all candidate answers base ONLY the provided context.

## Task
Generate a list of distinct, plausible answer candidates for the given question based on the provided context.

## Instructions
1. Analyze the question and the context carefully. Determine the question type from the question text.
2. If the provided context DOES NOT contain sufficient information to answer the question, you MUST return exactly ONE candidate object with "short_answer": "Insufficient information" and explain why in the "long_answer". Do not guess or use outside knowledge.
3. If information is present, formulate at least 1 and up to {max_candidates} distinct candidates.
4. Each candidate must follow the JSON format with 'short_answer' and 'long_answer'.
5. For 'short_answer', strictly follow these rules based on the question type:
   * For questions asking "how many" or about counts, return only a number or numeric phrase (e.g., "10", "ten").
   * For questions asking about time or duration, return a concise time value (e.g., "36 hours", "3 days").
   * For yes/no questions, return "Yes" or "No".
   * For questions starting with "Who", "Which", or "What" that ask for a specific entity (e.g., person, disease, drug, gene, chromosome), return the full noun phrase naming that entity.
     - Examples: "chromosome X", "interleukin-6", "dopamine receptor D2".
   * For questions asking about organism, return organism type (e.g., "bacterium", "fungus").
6. For 'long_answer': The context is concatenated from different references. The format MUST be: "Reference: [reference 1], [reference 2]. Explanation: [Detailed explanation based on the information]".

## Constraints
- MUST: Return a JSON array of objects.
- MUST: Generate up to {max_candidates} candidates (or exactly 1 if insufficient information).
- MUST NOT: Duplicate short answers conceptually or semantically, even if they are supported by different references.
- MUST NOT: Include conversational fillers.
- MUST NOT: Add extra parenthetical information in the short_answer.
- MUST NOT: Use markdown formatting (e.g., ```json or ```). Return RAW, parsable JSON only.

## Output Format (JSON)
[
    {{ 
        "short_answer": ""short answer as per rules or 'Insufficient information'" if the answer not found in the provide context",
        "long_answer": "Reference: [reference 1 from context], [reference 2 from context], ... Explanation: [Detailed explanation based on the information]"
    }}
]
"""
    user_content = f"""Information:
Question: {question}
Subquery: {search_kw}
Key entities: {key_entities}
Context: {context}

Description of Input Fields:
- Question: The main biomedical question to answer.
- Subquery: Search keywords extract from the main question, use to retrieve the Context.
- Key entities: Relevant important entities extracted from the question.
- Context: Text passages containing potential evidence.

## Task
Provide answer candidates for the question in a valid JSON array.
"""
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    return messages

if __name__ == "__main__":
    threads = [0, 500]
    BASE_RES = "medhop_results_llm_create_mcq/all_ans"
    metadata = load_json('data/metadata_medhop.json')
    
    extractor_model = spacy.load("en_ner_bc5cdr_md")
    extractor_model_2 = extractor_model_3 = None
    target_label = ["DISEASE", "CHEMICAL"]
    cnt = 0
    for idx, (qidx, sample) in enumerate(tqdm(metadata.items(), desc="Processing")):
        if not (threads[0] <= idx < threads[1]):
            continue 
        
        if qidx != '695539': #'686876':
            continue
        
        target_ans_type = metadata[qidx]['target_answer_type']
        check_yn = target_ans_type == 'correctness'
        type_q = metadata[qidx]['type_q'] 
        if check_yn or type_q != "direct":
            continue
        else:
            cnt += 1
        
        res_file_sample = f"{BASE_RES}/R4_step_3b_all_ans_{qidx}.json"
        
        if os.path.exists(res_file_sample):
            print(res_file_sample)
            
            res_data = load_json(res_file_sample)
            # if 'create_mcq' in res_data:
            #     continue
            
            question = res_data["1"]["final_qi"]["question"]
            search_kw = res_data["1"]["final_qi"]["search_keyword"]
            all_src_cand = res_data["1"]["final_qi"]["candidate"]
            
            all_src_ctx = ""
            all_snippet = ""
            
            for item in all_src_cand:
                if item['source'] in ['google', 'wiki']:
                    all_snippet += item['retrieval_information'] + " "
                    
                all_src_ctx += item['retrieval_information'] + " \n "
            
            info_extract_ent = search_kw + " " + all_snippet
            key_entities = extract_entities_from_evidence(extractor_model, extractor_model_2, extractor_model_3, info_extract_ent, target_label)
            
            messages = get_multiple_candidate_answer_prompt(question, search_kw, key_entities, all_src_ctx)
            
            result_json = request_api(messages=messages, model='o3-mini')

            try:
                result_dict = json.loads(result_json)
            except Exception as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Raw response: {result_json}")
                result_dict = []
                
            res_data['create_mcq'] = result_dict
            
            print(result_dict)
            # save_json(res_data, res_file_sample)
            print("save: ", res_file_sample)
    print("run: ", cnt)