import sys
import json
import os
import spacy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.extract_ner import extract_entities_from_evidence
from src.gpt_request import request_api
from tqdm import tqdm


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
def prompt_gen_answer_medhop_correctness(question, search_kw, key_entities, context):
    prompt = f"""You are a biomedical expert. Your task is to answer a question base on ONLY the provided context
        ### INSTRUCTIONS:
        1. **Deconstruct the Question:**
        - Identify the core relationship (e.g., Disease <-> Gene).
        - Identify ALL specific constraints/modifiers (e.g., "with symptom X", "in elderly patients", "specifically for type 2").
        
        2. **Verify Full Coverage (CRITICAL STEP):**
        - Check if the Context covers **every single constraint** found in the question.
        - **Rule of Specificity:** If the question is specific (e.g., "Disease with Symptom X") but the context is general (only mentions "Disease" without confirming/denying Symptom X), you MUST conclude "Insufficient information".
        - Do NOT assume general facts apply to specific conditions unless explicitly stated.

        3. **Determine the Label:**
        - **"yes"**: Context explicitly supports the whole question including all specific constraints.
        - **"no"**: Context explicitly contradicts any part of the question.
        - **"insufficient information"**: Context mentions the entities but misses one or more specific constraints (e.g., context lacks info about the specific symptom, dosage, or population mentioned in the question).

        INPUT DATA:
        Question: {question}
        Subquery: {search_kw}
        Key entities: {key_entities}
        Context: {context}

        Description of Input Fields:
        - Question: The main biomedical question to answer.
        - Subquery: Search keywords extract from the main question, use to retrieve the Context.
        - Key entities: Relevant important entities extracted from the question.
        - Context: Text passages containing potential evidence.

        Response format (strict JSON):
        {{  
            "short_answer": "yes", "no" or "insufficient information",
            "long_answer": "Reference: [reference 1 from context], [reference 2 from context], ... Explanation: [Detailed explanation based on the information]"
        }}
        """
    return prompt

if __name__ == "__main__":

    threads = [125, 250]
    threads_str = f'_{threads[0]}_{threads[1]}' #để cho vào lưu tên file
    metadata = load_json('data/metadata_medhop.json')
    golden = load_json(r'data\all_answer_golden_56.json')

    source = 'merge_google_wiki' # 'wiki' #'google'
    s_type = 'kmean' #'hdbscan' #kmean
    num_cluster = '_7' # "_3"

    # cache_cluster = load_json(r'src\cluster\final\YN_output_cluster_kmean_euclidean_top5_7.json')
    
    # cache_cluster = load_json(fr'src\cluster\final\output_cluster_kmean_euclidean_top5{num_cluster}.json')
    res_file_sample = fr'src\cluster\final\YN_answer_{s_type}_{source}{num_cluster}{threads_str}.json'
    
    cache_cluster = load_json(r'src\cluster\final\direct_output_cluster_kmean_euclidean_top5_7.json')

    cache_search_kw = load_json(r'data\dict_searchkw_direct_YN.json')

    extractor_model = spacy.load("en_ner_bc5cdr_md")
    extractor_model_2 = extractor_model_3 = None
    target_label = ["DISEASE", "CHEMICAL"]
    
    BASE_WIKI = "data/cache/filter_wikipedia_300"
    BASE_WIKI_NER = "data/cache/filter_ner_wikipedia_300"
    
    print(res_file_sample)
    cnt = 0

    if os.path.exists(res_file_sample):
        res_data = load_json(res_file_sample)
    else:
        res_data = {}
    for idx, (qidx, sample) in enumerate(tqdm(metadata.items(), desc="Processing")):
        if qidx in res_data:
            continue
        
        if not (threads[0] <= idx < threads[1]):
            continue

        if qidx not in cache_cluster:
            continue

        target_ans_type = metadata[qidx]['target_answer_type']
        check_yn = target_ans_type == 'correctness'
        type_q = metadata[qidx]['type_q']

        if check_yn == False or type_q != "direct" or golden[qidx]['True'] == []:
            continue

        cnt += 1

        question = sample['original question']
        search_kw = cache_search_kw[qidx]

        key_entities = extract_entities_from_evidence(
            extractor_model,
            extractor_model_2,
            extractor_model_3,
            search_kw,
            target_label
        )

      
        # ===== Gen answer theo từng cluster =====
        res_data[qidx] = {}
        sample_cluster = cache_cluster[qidx]
        
        for cluster_label, item in sample_cluster.items():           
            # context = "\n".join(item['docs']) + "\n Reference: [CRAWL_WIKI]: " + "\n".join(item['crawl_doc'])
            context = "\n".join(item['docs']) 
            
            for idx, crawl_doc in enumerate(item['crawl_doc']):
                if item['source'][idx] == 'crawl_ner_wiki':
                    src = f"CRAWL_NER_WIKI_{cluster_label}_{idx}"
                elif item['source'][idx] == 'crawl_wiki':
                    src = f"CRAWL_WIKI_{cluster_label}_{idx}"
                    
                context += f"\n Reference: [{src}]: {crawl_doc}" 
                
            if context.strip() == "":
                continue

            prompt = prompt_gen_answer_medhop_correctness(
                question,
                search_kw,
                key_entities,
                context
            )

            messages = [
                {"role": "user", "content": prompt}
            ]

            result_json = request_api(messages=messages, model='o3-mini')

            try:
                result_dict = json.loads(result_json)
            except Exception as e:
                print("JSON parse error:", e)
                print(result_json)

            res_data[qidx][cluster_label] = {
                'llm': result_dict,
                'context': context
            }

        save_json(res_data, res_file_sample)
        
    save_json(res_data, res_file_sample)
    print(res_file_sample)
    print("Total processed:", cnt)