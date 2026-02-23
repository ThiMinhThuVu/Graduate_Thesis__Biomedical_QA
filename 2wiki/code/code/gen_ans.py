import sys
import json
import os
import spacy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.extract_ner import extract_entities_from_evidence
from src.gpt_request import request_api
from tqdm import tqdm


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        

def old_prompt_gen_answer_medhop_step_3b(question, search_kw, key_entities, context):
    prompt = f"""You are a biomedical expert. Your task is to answer a question base on ONLY the provided context

        Instructions:
        1. Analyze the Context carefully to answer the Question.
        2. Determine whether the Context contains sufficient information to answer the Question.

        CASE A: Answer Found In Context
        - If the Context DOES contain the answer:
            - short_answer: Strictly follow the formatting rules below based on question type.
            - long_answer: the context is concatenate from different reference. Long answer must have this format: "Reference: [reference from context]. Detailed explanation based on the information".

        CASE B: Insufficient Information
        - If the provided Context does NOT contain the specific information needed to answer the question:
            - short_answer: Return "Insufficient information".
            - long_answer: Do NOT hallucinate. Explain why the question can't be answer based on the providing context.

        Short Answer Formatting Rules (when answer is found):
        * For questions asking "how many" or about counts, return only a number or numeric phrase (e.g., "10", "ten").
        * For questions asking about time or duration, return a concise time value (e.g., "36 hours", "3 days").
        * For yes/no questions, return "Yes" or "No".
        * For questions starting with "Who", "Which", or "What" that ask for a specific entity (e.g., person, disease, drug, gene, chromosome), return the full noun phrase naming that entity.
            - Examples: "chromosome X", "interleukin-6", "dopamine receptor D2".
        * For questions asking about organism, return organism type (e.g., "bacterium", "fungus").
        
        3. Do not add any extra commentary or text outside the JSON.

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
            "short_answer": "short answer as per rules or 'Insufficient information'" if the answer not found in the provide context,
            "long_answer": "Reference: [reference 1 from context], [reference 2 from context], ... Explanation: [Detailed explanation based on the information]"
        }}
        """
    return prompt

if __name__ == "__main__":

    threads = [0, 1000]
 
    metadata = load_json('data/metadata_medhop.json')
    golden = load_json(r'data\all_answer_golden_56.json')
    cache_cluster = load_json(r'src\cluster\cache\cluster_results_300_3.json')
    cache_search_kw = load_json(r'data\dict_searchkw_direct_WH.json')

    extractor_model = spacy.load("en_ner_bc5cdr_md")
    extractor_model_2 = extractor_model_3 = None
    target_label = ["DISEASE", "CHEMICAL"]

    BASE_WIKI = "data/cache/filter_wikipedia_300"
    BASE_WIKI_NER = "data/cache/filter_ner_wikipedia_300"
    res_file_sample = r'src\cluster\cache\answer_k_mean_300_3.json'

    cnt = 0
    # res_data = {}
    res_data = load_json(res_file_sample)

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

        if check_yn or type_q != "direct" or golden[qidx]['True'] == []:
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

        labels = cache_cluster[qidx]["labels"]
        docs = cache_cluster[qidx]["docs"]

        # ===== Gom doc theo cluster =====
        sample_cluster = {}

        for i, label in enumerate(labels):
            label = str(label)

            if label not in sample_cluster:
                sample_cluster[label] = ""

            hash_id = docs[i]["hashid"]
            url = docs[i]["url"]
            source = docs[i]["source"]

            if source == "crawl_wiki":
                file_path = f"{BASE_WIKI}/{hash_id}"
            else:
                file_path = f"{BASE_WIKI_NER}/{hash_id}"

            if not os.path.exists(file_path):
                continue

            filter_text = load_json(file_path)

            content = filter_text.get("crawl_wiki", {}).get("filter", {}).get(url, {}).get("tfidf", None)

            if content:
                sample_cluster[label] += "\n" + content

        # ===== Gen answer theo từng cluster =====
        res_data[qidx] = {}

        for cluster_label, context in sample_cluster.items():

            if context.strip() == "":
                continue

            prompt = old_prompt_gen_answer_medhop_step_3b(
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