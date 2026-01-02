import os
import spacy
import pickle
from tqdm import tqdm
from src.decomposition import *
from src.retrieval import *
from src.extract_ner import *
from src.generate_answer import *
from utils.norm_wikidata_pair import *
from utils.util import *
from classification.classification import load_model, classify_question
import pandas as pd

global wiki_cache_data
global nlp_bio
global nlp_standard
wiki_cache_data = {}
nlp_bio = None
nlp_standard = None

def load_spacy_model():
    global wiki_cache_data
    global nlp_bio
    global nlp_standard

    with open('data/wikipedia_cache.pkl', 'rb') as f:
        wiki_cache_data = pickle.load(f)
        
    try:
        nlp_bio = spacy.load("en_ner_bc5cdr_md")  # Biomedical NER model
        print("Loaded biomedical NER model")
    except:
        print("Biomedical NER model not found. Please install it with: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz")
        nlp_bio = None

    nlp_standard = spacy.load("en_core_web_lg")  # Standard English model

def generate_answer_for_subq(subquestion, key_entities, retrieval_information, target_label, step, paths, settings, dataset=None, type_q=None):
    ner_filepath = paths['ner']
    output_ner = load_json(ner_filepath) if os.path.exists(ner_filepath) else {}
    
    if step in output_ner:
        list_entities = output_ner[step]
    else:
        print(f"Extracting entities.")
        query_for_ner = subquestion + " " + retrieval_information
        list_entities = extract_with_models(query_for_ner, nlp_bio, nlp_standard, target_label, dataset=dataset)
        output_ner[step] = list_entities
        save_json(output_ner, ner_filepath)
   
    all_key_entities = ""
    for item in key_entities:
        all_key_entities += f"{item}, "
    
    all_key_entities += f"{target_label} Entities: " + ", ".join(list_entities)
    
    return generate_answer_new(subquestion, all_key_entities, retrieval_information, dataset=dataset, type_q=type_q)

def preprocess_sample(sample, settings, is_seq=False, dataset=None, type_q=None):
    sample_id = str(sample['QIDX']) 
    paths = setup_paths(settings, sample_id)
    file_all_ans = paths['final_answer']
    # evidence_file = paths['evidence']
    
    if os.path.exists(file_all_ans):
        print(f"All answers file already exists: {file_all_ans}")
        return
    
    # if os.path.exists(file_all_ans):
    #     dict_ans = load_json(file_all_ans)
    #     evidence_predict = gen_evidences(sample, dict_ans, type_q)
    #     save_json(evidence_predict, evidence_file)
        
    #     dict_ans['evidence'] = evidence_predict
    #     save_json(dict_ans, file_all_ans)
    #     print(f"Evidence saved to {evidence_file}")
    #     return 
    
    if dataset == "medhop" and sample_id in settings['chrunk_mapping']:
        original_question = settings['chrunk_mapping'][sample_id]
    else:
        original_question = sample['Question']
        
    #DECOMPOSITION Question
    list_qi = question_decomposition(original_question, dataset=dataset, type_q=type_q)
    n_qi = len(list_qi)
    
    dict_ans = {
        "QIDX": sample_id,
        "original_question": original_question
    }
    all_context = ""
    
    for q_i, question in enumerate(list_qi, start=1):
        qi_number = f"/{q_i}"
        paths = setup_paths(settings, sample_id, qi_number)
        ans_filepath = paths['qi_answer']
        dict_ans[q_i] = {
            "final_qi": {
                "question": "",
                "search_keyword": "",
                "key_entity": "",
                "short_answer": "",
                "long_answer": "",
                "retrieval_information": "",
                "prefix": "",
                "name": "",
                "suffix": "",
                "normalized_v2": ""
            }
        }
    
        path_subq_output, n_path = generate_subquestion(question, sample_id, paths['decomp'], is_seq=is_seq, dataset=dataset, type_q=type_q, model="gpt-4o-mini")
        
        for path_i in range(n_path):
            path_i = str(path_i)
            subq_output = path_subq_output[path_i]
            dict_ans[q_i][path_i] = {}
            qi_final_answers = {}
            check_path_i = False
            normalized_v2 = None
            if os.path.exists(ans_filepath):
                qi_final_answers = load_json(ans_filepath)
                if path_i in qi_final_answers:
                    check_path_i = True
                    
            if check_path_i:
                short_ans = ""
                long_ans = ""
                for step, item in subq_output.items():
                    step = str(step)   
                    subquestion = item['subquestion']
                    search_kw_subq = item['search_keyword'] + f" {short_ans}"
                    search_kw_subq_retrieval = search_kw_subq + f" {short_ans}"
                    target_labels = item['target_labels']
                    
                    if step == "0":
                        key_entities = item['key_entity']   
                    else:
                        key_entities = [short_ans]
                                            
                    retrieval_information = perform_retrieval(sample, key_entities, search_kw_subq_retrieval, step, paths, settings)

                    if step not in qi_final_answers[path_i]:
                        continue
                    short_ans = qi_final_answers[path_i][step]["short_answer"]
                    long_ans = qi_final_answers[path_i][step]["long_answer"]     
                    if 'candidate' in qi_final_answers[path_i][step]:
                        candidates = qi_final_answers[path_i][step]["candidate"] 
                    else:
                        candidates = []
                    if step not in dict_ans[q_i][path_i]:
                        dict_ans[q_i][path_i][step] = {}
                        
                    dict_ans[q_i][path_i][step]["question"] = subquestion
                    dict_ans[q_i][path_i][step]["search_keyword"] = search_kw_subq
                    dict_ans[q_i][path_i][step]["key_entity"] = key_entities
                    dict_ans[q_i][path_i][step]["target_labels"] = target_labels
                    dict_ans[q_i][path_i][step]["short_answer"] = short_ans
                    dict_ans[q_i][path_i][step]["long_answer"] = long_ans
                    dict_ans[q_i][path_i][step]["retrieval_information"] = retrieval_information
                    dict_ans[q_i][path_i][step]["candidate"] = candidates
                    dict_ans[q_i][path_i][step]['normalized_v2'] = normalized_v2                                
            else:
                short_ans = ""
                qi_final_answers[path_i] = {}
                
                for step, item in subq_output.items():
                    step = str(step)
                    subquestion = item['subquestion']
                    search_kw_subq = item['search_keyword'] 
                    search_kw_subq_retrieval = search_kw_subq + f" {short_ans}"
                    target_labels = item['target_labels']
                    
                    if step == "0":
                        key_entities = item['key_entity']   
                    else:
                        key_entities = [short_ans]
                
                    print(f"\n--- Processing Step {step}: {subquestion} ---")
                    
                    # Multiple Choice
                    candidates = []
                    idx_del = 0
                    max_candidates = 10
                    current_retrieval = perform_retrieval(sample, key_entities, search_kw_subq_retrieval, step, paths, settings)
                    while idx_del < max_candidates:
                        if current_retrieval == "" or current_retrieval == []:
                            break
                            
                        ans_gen = generate_answer_for_subq(subquestion, key_entities, current_retrieval, target_labels, step, paths, settings, dataset=dataset, type_q=type_q)
                        
                        current_short = ans_gen["short_answer"]
                        print("Current answer: ", current_short)
                        # Kiểm tra nếu đáp án mới này đã tồn tại trong list candidates chưa
                        is_duplicate = any(c['short_answer'].lower().strip() == current_short.lower().strip() for c in candidates)
                        
                        if not is_duplicate and current_short.lower() != "insufficient information":
                            candidates.append({
                                "short_answer": current_short,
                                "long_answer": ans_gen["long_answer"],
                                "retrieval_information": current_retrieval,
                                "full_data": ans_gen # Lưu lại để dùng sau khi chọn
                            })
                        else:
                            break
                        
                        if current_short.lower() in ['yes', 'no']:
                            break
                        
                        new_retrieval = remove_perform_retrieval(current_retrieval, current_short)
                        if new_retrieval == current_retrieval:
                            break
                        else:
                            current_retrieval = new_retrieval
                        idx_del += 1
                    
                    # --- CHỌN ĐÁP ÁN CUỐI CÙNG TỪ CÁC CANDIDATES ---
                    if len(candidates) > 1:
                        # Build context cho Multiple Choice
                        mc_options_text = ""
                        for i, cand in enumerate(candidates):
                            label = chr(ord('A') + i)
                            mc_options_text += f"{label}. {cand['short_answer']}\nReason: {cand['long_answer']}\n\n"
                        try:
                            selected_label = gen_ans_multiple_choice(subquestion, mc_options_text, dataset=dataset)
                            
                            idx_selected = ord(selected_label[0].upper()) - ord('A')
                            if 0 <= idx_selected < len(candidates):
                                final_candidate = candidates[idx_selected]
                                answer_generation = final_candidate["full_data"]
                                retrieval_information = final_candidate["retrieval_information"]
                            else:
                                answer_generation = candidates[0]["full_data"]
                                retrieval_information = candidates[0]["retrieval_information"]
                        except Exception as e:
                            print(f"Error in MC selection: {e}")
                            answer_generation = candidates[0]["full_data"]
                            retrieval_information = candidates[0]["retrieval_information"]
                            
                    elif len(candidates) == 1:
                        answer_generation = candidates[0]["full_data"]
                        retrieval_information = candidates[0]["retrieval_information"]
                    else:
                        # Trường hợp không tìm thấy gì (insufficient)
                        answer_generation = {"short_answer": "insufficient information", "long_answer": "No snippets found."}
                        retrieval_information = []               
                           
                    short_ans = answer_generation["short_answer"]
                    long_ans = answer_generation["long_answer"]
    
                    if type_q == "inference":
                        prefix_ans = answer_generation['prefix']
                        name_ans = answer_generation['name']
                        suffix_ans = answer_generation['suffix']

                    qi_final_answers[path_i][step] = answer_generation
                
                    print(f"Short Answer for Step {step}: {short_ans}")

                    dict_ans[q_i][path_i][step] = {} 
                    dict_ans[q_i][path_i][step]["question"] = subquestion
                    dict_ans[q_i][path_i][step]["search_keyword"] = search_kw_subq
                    dict_ans[q_i][path_i][step]["key_entity"] = key_entities
                    dict_ans[q_i][path_i][step]["target_labels"] = target_labels
                    dict_ans[q_i][path_i][step]["short_answer"] = short_ans
                    dict_ans[q_i][path_i][step]["long_answer"] = long_ans
                    dict_ans[q_i][path_i][step]["candidate"] = candidates
                    dict_ans[q_i][path_i][step]["retrieval_information"] = retrieval_information
                    if type_q == "inference":
                        dict_ans[q_i][path_i][step]['prefix'] = prefix_ans
                        dict_ans[q_i][path_i][step]['name'] = name_ans
                        dict_ans[q_i][path_i][step]['suffix'] = suffix_ans
                    
                    if short_ans.lower() == "insufficient information": # sang path tiếp theo 
                        break
            
            # Lấy output step cuối cùng của 1 path duy nhất của qi:
            if short_ans.lower() != "insufficient information":
                dict_ans[q_i]['final_qi']["question"] = question
                dict_ans[q_i]['final_qi']["search_keyword"] = search_kw_subq
                dict_ans[q_i]['final_qi']["key_entity"] = key_entities
                dict_ans[q_i]['final_qi']["target_labels"] = target_labels
                dict_ans[q_i]['final_qi']["short_answer"] = short_ans
                dict_ans[q_i]['final_qi']["long_answer"] = long_ans
                dict_ans[q_i]['final_qi']["retrieval_information"] = retrieval_information
                dict_ans[q_i][path_i][step]["candidate"] = candidates
                dict_ans[q_i]['final_qi']['normalized_v2'] = normalized_v2
                if type_q == "inference":
                    dict_ans[q_i]['final_qi']['prefix'] = prefix_ans
                    dict_ans[q_i]['final_qi']['name'] = name_ans
                    dict_ans[q_i]['final_qi']['suffix'] = suffix_ans   

                    # For 'inference' questions, the first valid path is sufficient.
                    break
        
        all_context += f"{q_i}.{question} - {dict_ans[q_i]['final_qi']['short_answer']}, {dict_ans[q_i]['final_qi']['long_answer']}\n\n"
        save_json(qi_final_answers, ans_filepath)
        print(f"\nFinal answer saved to {ans_filepath}")
        
    if n_qi > 1:
        combine_parallel_ans = generate_combine_parallel_answers(original_question, all_context, dataset=dataset, model="o3-mini")
        dict_ans['final_answer'] = combine_parallel_ans
        save_json(dict_ans, file_all_ans)
        print(f"Final answer: {combine_parallel_ans['short_answer']}")
        print(f"\nSaved to {file_all_ans}")
    else:
        dict_ans['final_answer'] = {
            'long_answer': dict_ans[1]['final_qi']['long_answer'],
            'short_answer': dict_ans[1]['final_qi']['short_answer'],
            'retrieval_information': dict_ans[1]['final_qi']['retrieval_information']
        }
        if type_q == "inference":
            dict_ans['final_answer']['prefix'] = dict_ans[1]['final_qi']['prefix']
            dict_ans['final_answer']['name'] = dict_ans[1]['final_qi']['name']
            dict_ans['final_answer']['suffix'] = dict_ans[1]['final_qi']['suffix']
            dict_ans['final_answer']['normalized_v2'] = dict_ans[1]['final_qi']['normalized_v2']
            
        save_json(dict_ans, file_all_ans)
        print(f"Final answer: {dict_ans['final_answer']['short_answer']}")
        print(f"\nSaved to {file_all_ans}")
        
    # evidence_predict = gen_evidences(sample, dict_ans, type_q)
    # save_json(evidence_predict, evidence_file)
    
    # dict_ans['evidence'] = evidence_predict
    # save_json(dict_ans, file_all_ans)
    # print(f"Evidence saved to {evidence_file}")     

if __name__ == "__main__":
    config_path = "config/config.yaml"
    configs = load_yaml(config_path)
    
    #edit config
    threads = configs['run']['threads'] 
    dataset = configs['run']['dataset']
    data_type = configs['run']['data_split']  
    
    model_path = configs['classification'][dataset]['model_path']
    tokenizer_dir = configs['classification'][dataset]['tokenizer_dir']

    model, tokenizer, device = load_model(dataset, model_path, tokenizer_dir)
    
    settings = configs['type_run']   
    settings["dataset"] = f"{dataset}_"
    settings["wiki_cache"] = wiki_cache_data
    raw_data_file = configs['dataset'][dataset][data_type]
    raw_data = load_json(raw_data_file)
        
    if dataset == "medhop":
        settings['chrunk_mapping'] = load_json(r'data\raw\medhop\shrunk_mapping.json')
        
    load_spacy_model()
                  
    for idx, (ID, sample) in enumerate(tqdm(raw_data.items(), desc="Processing")):
        if not (threads[0] <= idx < threads[1]):
            continue 
        question = sample['Question']
        type_q, probabilities = classify_question(dataset, model, tokenizer, device, question)
        if type_q in ['non-sequence', 'comparison']:
            is_seq = False
        else:
            is_seq = True
    
        preprocess_sample(sample, settings, is_seq=is_seq, dataset=dataset, type_q=type_q)
