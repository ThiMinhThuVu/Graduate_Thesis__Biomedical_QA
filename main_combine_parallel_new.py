import os
import spacy
from tqdm import tqdm
from src.decomposition import decompose_question_parallel, decompose_question_sequential
from src.gpt_request import  request_api
from utils.util import *
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

import shutil

global hash_id_cache
global hash_id_crawl_wiki_cache
global extractor_model
global extractor_model_2 
global extractor_model_3
global embedding_model
global embedding_model_cluster
extractor_model = None

global cache_ner_filter
cache_ner_filter = {}
mapping_data_answer = load_json('data/mapping_answer_gol.json')

dict_golden = load_json(r'data\all_answer_golden_56_update.json')

def reset_directory(dir_path: str):
    """
    Delete the directory and all its contents,
    then recreate an empty directory.
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    
def load_model(dataset, configs, using_embedding_model=False):
    global hash_id_cache
    global hash_id_crawl_wiki_cache
    global extractor_model
    global extractor_model_2
    global extractor_model_3
    global embedding_model
    global embedding_model_cluster


    print(configs['cache'])
    hash_id_cache = load_json(configs['type_run']['cache']['hash_id']['retrieval'])
    hash_id_crawl_wiki_cache = load_json(configs['type_run']['cache']['hash_id']['crawl_wiki'])
    
    if dataset == "medhop":
        # Biomedical NER
        extractor_model = spacy.load("en_ner_bc5cdr_md")
        extractor_model_2 = spacy.load("en_ner_jnlpba_md")
        extractor_model_3 = spacy.load("en_ner_bionlp13cg_md")
        # extractor_model_2 = extractor_model_3 = None
      
        embedding_model_cluster = SentenceTransformer(
            "pritamdeka/S-BioBert-snli-multinli-stsb",
            device='cpu'
        )
        if using_embedding_model:
            # Initialize a StaticEmbedding module
            static = StaticEmbedding.from_model2vec("neuml/pubmedbert-base-embeddings-8M")
            embedding_model = SentenceTransformer(modules=[static])
        else:
            embedding_model = None
        print("Loaded biomedical NER + embedding model")

    elif dataset == "wiki":
        extractor_model = spacy.load("en_core_web_lg")
        extractor_model_2 = extractor_model_3 = None
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Loaded general NER + embedding model")

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

import json

def generate_combine_parallel_answers_new(original_question, all_context, model="o3-mini"):
    """
    Prompt đã được khử dữ liệu tập test, sử dụng ví dụ trung lập 
    để hướng dẫn logic mapping và negative constraint.
    """
    # đúng 6/12 - v2
    # prompt = f"""## Role
    #     You are a Medical Logic Auditor and Entity Resolver. Your goal is to synthesize multi-hop reasoning into a final answer that adheres to strict formatting rules.

    #     ## Task
    #     1. Analyze the 'Root Question' to determine its type (Choice or Confirmation).
    #     2. Review the 'Reasoning Hops' to extract evidence.
    #     3. Perform 'Entity Resolution' to map Boolean results (Yes/No) back to the entities in the Root Question.

    #     ## Instructions
    #     ### Step 1: Question Type Detection
    #     - **Choice Questions**: If the question contains "A or B", "X vs Y", "Normal or Abnormal", it is a CHOICE question.
    #     - **Confirmation Questions**: If the question asks "Is...", "Are...", "Do...", it is a YES/NO question.

    #     ### Step 2: Answer Normalization & Mapping (Literal Phrase Rule)
    #     - **For Choice Questions**: 
    #         - Identify the options provided in the Root Question (e.g., "A" or "B").
    #         - **MANDATORY**: You MUST extract the chosen option **EXACTLY** as it appears in the Root Question text. 
    #         - **ANTI-TRUNCATION**: Do NOT shorten, simplify, or paraphrase the entity. If the question says "normal testosterone levels", do NOT return "normal". You must return the full phrase.
    #         - **Literal Matching**: The `short_answer` must be a sub-string of the Root Question.
            
    #     ### Step 3: Negative Constraint & Conflict Audit
    #     - **Conflict Resolution**: If Hop 1 says "Yes" for Option A and Hop 2 says "Yes" for Option B, but the question is a "Which one" type, this is a CONFLICT. 
    #     - In case of conflict: Re-read the context to see which entity is linked to the EXACT criteria mentioned in the Root Question.
    #     - **Association Audit**: Remember, being "mentioned" is not the same as being "the answer".
        
    #     ## Information
    #     Root Question: {original_question}
    #     Reasoning Hops Context:
    #     {all_context}

    #     ## Output Format (JSON)
    #     {{
    #         "question_type_detected": "choice | confirmation",
    #         "audit_logic": "Explain the mapping logic and check for any clinical exceptions (Negative Constraints)",
    #         "long_answer": "A cohesive clinical explanation synthesized from hops",
    #         "short_answer": "The final normalized answer (Exact option string for choice, or 'yes'/'no')"
    #     }}
    #     """
    
    # version 3: no-short ans: 7/12
    # prompt = f"""## Role
    #     You are a Medical Logic Auditor and Entity Resolver. Your goal is to synthesize multi-hop reasoning into a final answer that adheres to strict formatting rules.

    #     ## Task
    #     1. Analyze the 'Root Question' to determine its type (Choice or Confirmation).
    #     2. Review the 'Reasoning Hops' to extract evidence.
    #     3. Perform 'Entity Resolution' to map Boolean results (Yes/No) back to the entities in the Root Question.

    #     ## Instructions
    #     ### Step 1: Question Type Detection (Internal Reasoning)
    #     - **Choice Questions**: If the question contains "A or B", "X vs Y", "Normal or Abnormal", or asks to identify a specific condition among options.
    #     - **Confirmation Questions**: If the question asks "Is...", "Are...", "Do...".

    #     ### Step 2: Answer Normalization & Mapping (Literal Phrase Rule)
    #     - **For Choice Questions**: 
    #         - Identify the options provided in the Root Question (e.g., "normal testosterone levels" or "abnormal testosterone levels").
    #         - **MANDATORY**: You MUST extract the chosen option **EXACTLY** as it appears in the Root Question text. 
    #         - **ANTI-TRUNCATION**: Do NOT shorten or simplify the entity. If the question says "normal testosterone levels", do NOT return "normal". 
    #         - **Literal Matching**: The `short_answer` for a Choice Question must be a literal sub-string of the Root Question.

    #     ### Step 3: Conflict & Negative Constraint Audit
    #     - **Exclusivity**: If the question asks for a singular condition but hops provide different conditions for different parts, identify if there is a common intersection. If no single condition meets all criteria, the `short_answer` should reflect that (e.g., "none" or listing both clearly as a sub-string).
    #     - **Association Check**: Distinguish "first described by" from "clinically associated with". Historical mention is NOT a clinical association.

    #     ## Rules
    #     - Provide ONLY the JSON object.
    #     - No commentary or preamble.
    #     - `short_answer` for Confirmation: "yes" or "no" (lowercase).
    #     - `short_answer` for Choice: The exact string from the question.

    #     ## Information
    #     Root Question: {original_question}
    #     Reasoning Hops Context:
    #     {all_context}

    #     ## Output Format (JSON)
    #     {{
    #         "long_answer": "A cohesive clinical explanation synthesized from hops, justifying the mapping and addressing any potential exceptions.",
    #         "short_answer": "The final normalized answer string"
    #     }}
    #     """
    
    prompt = f"""## Role
You are a Medical Logic Auditor and Entity Resolver. Your goal is to synthesize multi-hop reasoning into a final answer that adheres to strict formatting rules.

## Task
1. Analyze the 'Root Question' to determine its type: Choice, Confirmation, or Identification (Commonality).
2. Review the 'Reasoning Hops' to extract evidence.
3. If the question asks about multiple mutations/conditions (A and B), find the **highest-level common clinical entity** that covers both.

## Instructions
### Step 1: Question Type Detection
- **Choice**: Contains "A or B", "X vs Y". Return the exact string from the question.
- **Confirmation**: "Is...", "Are...", "Do...". Return "yes" or "no".
- **Identification (Commonality)**: "What condition is associated with A AND B?". Find the shared medical category (e.g., if A is LGMD2A and B is LGMD2I, the common condition is "limb-girdle muscular dystrophy").

### Step 2: Answer Normalization
- **Commonality Rule**: If multiple hops return specific subtypes (e.g., "type 2A", "type 2I"), the `short_answer` MUST be the **common denominator** (e.g., "limb-girdle muscular dystrophy").
- **Literal Matching (For Choice)**: The `short_answer` must be a literal sub-string of the Root Question.
- **Precision**: Do not include citations ([GG_1]) in the `short_answer`.

### Step 3: Conflict Audit
- If the hops confirm two different specific conditions but they belong to the same family, name the family. 
- Ensure the `long_answer` explains how both specific mutations lead to the general condition.

## Rules
- Provide ONLY the JSON object.
- No commentary, no markdown code blocks, no preamble.
- JSON keys must be "long_answer" and "short_answer".

## Information
Root Question: {original_question}
Reasoning Hops Context:
{all_context}

## Output Format (JSON)
{{
    "long_answer": "...",
    "short_answer": "..."
}}
"""
                
    messages = [
        {
            "role": "system", 
            "content": "You are an expert medical auditor. You MUST output ONLY a valid JSON object. No conversational filler."
        },
        {"role": "user", 
         "content": prompt
        }
    ]
    
    result_json = request_api(messages, model)
    
    try:
        clean_json = result_json.replace("```json", "").replace("```", "").strip()
        result_dict = json.loads(clean_json)
        
        return {
            "long_answer": result_dict.get("long_answer", ""),
            "short_answer": result_dict.get("short_answer", ""),
        }
        
    except Exception as e:
        return {"long_answer": result_json, "short_answer": "Error combine"}

def preprocess_sample(list_ner_q, sample, settings, is_seq=False, dataset=None, type_q=None,
            using_target_type = False, using_gen_old=False, has_extract_ner=False, using_decomp_new=False,
            question_type_to_ner=None, using_prompt_yn=False, file_cache_o3_zeroshot=None, KB=""):
    
    sample_id = str(sample['QIDX']) 
    print("[ID]: ", sample_id)
    paths = setup_paths(KB, settings, sample_id)
    file_all_ans = paths['final_answer']
    cache_ner_filter[sample_id] = {}
    
    # if os.path.exists(file_all_ans):
    #     return
    
    if dataset == "medhop" and sample_id in settings['chrunk_mapping']:
        original_question = settings['chrunk_mapping'][sample_id]
    else:
        original_question = sample['Question']
        
    decomp_file = paths['cache_decomp']
    if os.path.exists(decomp_file):
        list_qi = load_json(decomp_file)
    else:
        list_qi = decompose_question_parallel(original_question, dataset=dataset, type_q=type_q)
        save_json(list_qi, decomp_file)
        
    n_qi = len(list_qi)                            

    if os.path.exists(file_all_ans):
        dict_ans = load_json(file_all_ans)
    else:
        dict_ans = {
            "QIDX": sample_id,
            "original_question": original_question,
            "type_q": sample['type_q'],
            "key_entities": list_ner_q
        }

    all_context = ""
    
    history_evidence = load_json('history_evidence.json')
    if sample_id in history_evidence:
        return 
    else:
        history_evidence[sample_id] = {}
    for q_i, question in enumerate(list_qi, start=1):
        q_i = str(q_i)

        short_ans = dict_ans[q_i]['final_qi']["short_answer"]
        long_ans = dict_ans[q_i]['final_qi']["long_answer"]
                 
        # all_context += f"Hop {q_i}:{question} - Explanation: {long_ans}\n\n"
        all_context += f"Hop {q_i}:{question} - Answer:{short_ans} -Explanation: {long_ans}\n\n"
        
        history_evidence[sample_id][q_i] = {
            'question': question,
            'short_answer': short_ans,
            'long_answer': long_ans
        }
        
    history_evidence[sample_id]['original_question'] = original_question
    if n_qi > 1:
        # combine_parallel_ans = generate_combine_parallel_answers(original_question, all_context, dataset=dataset, model="o3-mini")
        combine_parallel_ans = generate_combine_parallel_answers_new(original_question, all_context, model="o3-mini")
        history_evidence[sample_id]['final_answer'] = combine_parallel_ans
        short_ans_final = combine_parallel_ans['short_answer']
        # save_json(history_evidence,'history_evidence.json')
        # print(f"Final answer: {combine_parallel_ans['short_answer']}")
    else:
        history_evidence[sample_id]['final_answer'] = {
            'short_answer': short_ans,
            'long_answer': long_ans
        }
        short_ans_final = short_ans
        
    if short_ans_final.lower() in dict_golden[sample_id]['True']:
        history_evidence[sample_id]['final_answer']['check'] = True
    else:
        history_evidence[sample_id]['final_answer']['check'] = False
        
    history_evidence[sample_id]['final_answer']['golden'] =  dict_golden[sample_id]['True']
    
    print(f"Final answer: {history_evidence[sample_id]['final_answer']['short_answer']}")
    save_json(history_evidence,'history_evidence.json')
     

if __name__ == "__main__":
    config_path = "config/config.yaml"
    configs = load_yaml(config_path)
    
    #edit config
    KB="PARALLEL_1000_LITE" # Answer2, ...
    threads = [0, 1000] 
    # configs['run']['threads'] 
    dataset = configs['run']['dataset']
    data_type = configs['run']['data_split']  
    
    model_path = configs['classification'][dataset]['model_path']
    tokenizer_dir = configs['classification'][dataset]['tokenizer_dir']
    
    settings = configs['type_run']   
    settings["dataset"] = f"{dataset}_"
    raw_data_file = configs['dataset'][dataset][data_type]
    raw_data = load_json(raw_data_file)
        
    if dataset == "medhop":
        settings['chrunk_mapping'] = load_json(configs['dataset'][dataset]['chrunk_mapping'])
        settings['normalize'] = True
        
    #EDIT CONFIG VERSION NER
    # reset_directory(r"data/cache/filter_ner_wikipedia")
    # reset_directory(r"data/cache/filter_wikipedia")
    load_model(dataset, configs, using_embedding_model=False) #False = tfidf, true = embedding_model
    # ner_version = 4
    ner_version = "#3"
    metadata = load_json('data/metadata_medhop.json')
    meta_ner = load_json(r'data/metadata_entity.json') 
    cnt = 0
    
    par_wrong = load_json(r'analysis_parallel_1000.json')
    for idx, (ID, sample) in enumerate(tqdm(raw_data.items(), desc="Processing")):
        if not (threads[0] <= idx < threads[1]):
            continue 
        
        qidx = sample['QIDX']
        question = sample['Question']
        
        # if qidx not in par_wrong:
        #     continue
        
        target_ans_type = metadata[qidx]['target_answer_type']
        
        # CONFIG YESNO
        # check_yn = target_ans_type == 'correctness'
        # if check_yn:
        #     # continue
        # else:
        #     # continue
        
        type_q = metadata[qidx]['type_q']    
        
        #CONFIG    
        if type_q != "parallel": # or dict_golden[qidx]['True'] == []:# "sequence": # "parallel":
            continue
        else:
            cnt += 1
    
        sample['type_q'] = type_q        
        if type_q in ['direct', 'parallel', 'comparison']:
            is_seq = False
        else:
            is_seq = True

        if ner_version == 2:
            question_type_to_ner = question_type_to_ner_2
        elif ner_version == 3:
            question_type_to_ner = question_type_to_ner_3
        elif ner_version == 4:
            question_type_to_ner = question_type_to_ner_4
        elif ner_version == "#3":
            question_type_to_ner = {} # để tất cả các type trả về là ["DISEASE", "CHEMICAL"]   
        
        dict_ner_q = meta_ner[qidx]['entities_q'].get('detail', {})
        list_ner_q = list(dict_ner_q.keys())
        
        file_cache_o3_zeroshot = configs['cache']['zeroshot'][type_q] # direct, sequence, parallel
        
        preprocess_sample(list_ner_q, sample, settings, is_seq=is_seq, dataset=dataset, type_q=type_q, 
            using_target_type=False, using_gen_old=True, has_extract_ner=True, using_decomp_new=True,
            question_type_to_ner=question_type_to_ner, file_cache_o3_zeroshot=file_cache_o3_zeroshot, KB=KB)
        
    print(cnt)
    