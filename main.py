import os
import spacy
from tqdm import tqdm
from src.decomposition import decompose_question_parallel, decompose_question_sequential
from src.retrieval import retrieve_evidence, remove_retrieve_evidence
from src.extract_ner import extract_entities_from_evidence
from src.generate_answer import generate_combine_parallel_answers,gen_ans_multiple_choice, _generate_answer
from utils.util import *
from classification.classification_run import load_model_classification, classify_question
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from transformers import AutoModel

global hash_id_cache
global wiki_cache_data
global extractor_model
global extractor_model_2 
global extractor_model_3
global embedding_model
wiki_cache_data = {}
extractor_model = None

def load_model(dataset, configs):
    global hash_id_cache
    global wiki_cache_data
    global extractor_model
    global extractor_model_2
    global extractor_model_3
    global embedding_model

    wiki_cache_data = load_pickle(configs['cache']['wiki_cache_data'])
    hash_id_cache = load_json(configs['cache']['hash_id_cache'])

    if dataset == "medhop":
        # Biomedical NER
        extractor_model = spacy.load("en_ner_bc5cdr_md")
        extractor_model_2 = spacy.load("en_ner_jnlpba_md")
        extractor_model_3 = spacy.load("en_ner_bionlp13cg_md")
        # extractor_model_2 = extractor_model_3 = None
        
        # Biomedical embedding
        # embedding_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        # embedding_model = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-base")  
        # embedding_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        
        # Initialize a StaticEmbedding module
        static = StaticEmbedding.from_model2vec("neuml/pubmedbert-base-embeddings-8M")
        embedding_model = SentenceTransformer(modules=[static])
        print("Loaded biomedical NER + embedding model")

    elif dataset == "wiki":
        extractor_model = spacy.load("en_core_web_lg")
        extractor_model_2 = extractor_model_3 = None
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Loaded general NER + embedding model")

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def preprocess_sample(sample, settings, is_seq=False, dataset=None, type_q=None, using_target_type = False):
    sample_id = str(sample['QIDX']) 
    paths = setup_paths(settings, sample_id)
    file_all_ans = paths['final_answer']
    
    if os.path.exists(file_all_ans):
        print(f"All answers file already exists: {file_all_ans}")
        return
        
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
    
    dict_ans = {
        "QIDX": sample_id,
        "original_question": original_question,
        "type_q": sample['type_q'],
        "probs": sample['probs'],
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
                "suffix": ""
            }
        }
    
        path_output, n_path = decompose_question_sequential(question, sample_id, paths['decomp'], is_seq=is_seq, dataset=dataset, type_q=type_q, model="gpt-4o-mini")
        # continue 
        
        for path_i in range(n_path):
            path_i = str(path_i)
            subq_output = path_output[path_i]
            dict_ans[q_i][path_i] = {}
            qi_final_answers = {}
            check_path_i = False
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
                    search_kw_subq = item['search_keyword']
                    search_kw_subq_retrieval = search_kw_subq + f" {short_ans}"
                    focus_spacy_based_entity_label = item['focus_spacy_based_entity_label']
                    target_answer_type = item.get('target_answer_type', 'other')
                    
                    if step == "0":
                        key_entities = item['key_entity']   
                    else:
                        key_entities = [short_ans]
                        
                    print(f"\n--- Load Step {step}: {subquestion} ---")
                                            
                    retrieval_information = retrieve_evidence(sample, key_entities, search_kw_subq_retrieval, settings, embedding_model, hash_id_cache)

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
                    dict_ans[q_i][path_i][step]["focus_spacy_based_entity_label"] = focus_spacy_based_entity_label
                    dict_ans[q_i][path_i][step]["target_answer_type"] = target_answer_type
                    dict_ans[q_i][path_i][step]["short_answer"] = short_ans
                    dict_ans[q_i][path_i][step]["long_answer"] = long_ans
                    dict_ans[q_i][path_i][step]["retrieval_information"] = retrieval_information
                    dict_ans[q_i][path_i][step]["candidate"] = candidates
            else:
                short_ans = ""
                qi_final_answers[path_i] = {}
                
                for step, item in subq_output.items():
                    step = str(step)
                    subquestion = item['subquestion']
                    search_kw_subq = item['search_keyword'] 
                    search_kw_subq_retrieval = search_kw_subq + f" {short_ans}"
                    focus_spacy_based_entity_label = item['focus_spacy_based_entity_label']
                    target_answer_type = item.get('target_answer_type', 'other')
                    
                    if step == "0":
                        key_entities = item['key_entity']   
                    else:
                        key_entities = [short_ans]
                
                    print(f"\n--- Processing Step {step}: {subquestion} ---")
                    
                    # Multiple Choice
                    candidates = []
                    idx_del = 0
                    current_retrieval, cache_filepath_retrieval = retrieve_evidence(sample, key_entities, search_kw_subq_retrieval, settings, embedding_model, hash_id_cache)
                    
                    # evidence_ner = extract_entities_from_evidence(extractor_model, extractor_model_2, extractor_model_3, current_retrieval, focus_spacy_based_entity_label, cache_filepath_retrieval, settings) 
                    # answer_generation = _generate_answer(subquestion, evidence_ner, current_retrieval, dataset, type_q, target_answer_type, using_target_type)
                    
                    while idx_del < settings['max_candidates']:
                        if current_retrieval == "" or current_retrieval == []:
                            break
                        
                        evidence_ner = extract_entities_from_evidence(extractor_model, extractor_model_2, extractor_model_3, current_retrieval, focus_spacy_based_entity_label, cache_filepath_retrieval, settings)  
                        # norm_rules = apply_answer_normalization_rules(question=subquestion,dataset=dataset,type_q=type_q)
                        # raw_answer = generate_answer(subquestion, target_answer_type, current_retrieval, norm_rules, q_key_entity=key_entities, evidence_key_entity=evidence_ner, dataset=dataset, type_q=type_q)
                        # không norm
                        normalized_answer = _generate_answer(subquestion, evidence_ner, current_retrieval, dataset, type_q, target_answer_type, using_target_type)
                        current_short = normalized_answer['short_answer']
                        
                        # # Normalize if needed
                        # if settings['normalize'] and norm_rules.get("needs_normalization", True):
                        #     normalized_answer = normalize_answer_with_wikipedia(
                        #         raw_answer,
                        #         should_normalize=True
                        #     )
                        # else:
                        #     normalized_answer = raw_answer.copy()
                        #     normalized_answer["normalization"] = {
                        #         "was_normalized": False,
                        #         "skip_reason": "Normalization disabled or not needed"
                        #     }
                        # normalized_answer["metadata"] = {
                        #     "rules_applied": norm_rules.get("primary_rule"),
                        #     "normalized": settings['normalize'] and norm_rules.get("needs_normalization", True),
                        # }
                                                
                        # current_short = normalized_answer["short_answer"]
                        # print("Current answer: ", current_short)
                        
                        # Kiểm tra nếu đáp án mới này đã tồn tại trong list candidates chưa
                        is_duplicate = any(c['short_answer'].lower().strip() == current_short.lower().strip() for c in candidates)
                        
                        if not is_duplicate and current_short.lower() != "insufficient information":
                            candidates.append({
                                "short_answer": current_short,
                                "long_answer": normalized_answer["long_answer"],
                                "retrieval_information": current_retrieval,
                                "full_data": normalized_answer # Lưu lại để dùng sau khi chọn
                            })
                        else:
                            break
                        
                        if current_short.lower() in ['yes', 'no']:
                            break
                        
                        new_retrieval = remove_retrieve_evidence(current_retrieval, current_short)
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
                    dict_ans[q_i][path_i][step]["focus_spacy_based_entity_label"] = focus_spacy_based_entity_label
                    dict_ans[q_i][path_i][step]["target_answer_type"] = target_answer_type
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
                dict_ans[q_i]['final_qi']["focus_spacy_based_entity_label"] = focus_spacy_based_entity_label
                dict_ans[q_i]['final_qi']["target_answer_type"] = target_answer_type
                dict_ans[q_i]['final_qi']["short_answer"] = short_ans
                dict_ans[q_i]['final_qi']["long_answer"] = long_ans
                dict_ans[q_i]['final_qi']["retrieval_information"] = retrieval_information
                dict_ans[q_i][path_i][step]["candidate"] = candidates
                if type_q == "inference":
                    dict_ans[q_i]['final_qi']['prefix'] = prefix_ans
                    dict_ans[q_i]['final_qi']['name'] = name_ans
                    dict_ans[q_i]['final_qi']['suffix'] = suffix_ans   

                    # For 'inference' questions, the first valid path is sufficient.
                    break
        
        all_context += f"{q_i}.{question} - {dict_ans[q_i]['final_qi']['short_answer']}, {dict_ans[q_i]['final_qi']['long_answer']}\n\n"
        save_json(qi_final_answers, ans_filepath)
        print(f"\nFinal answer saved to {ans_filepath}")
    
    # return
    
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
            
        save_json(dict_ans, file_all_ans)
        print(f"Final answer: {dict_ans['final_answer']['short_answer']}")
        print(f"\nSaved to {file_all_ans}")
    
    
if __name__ == "__main__":
    config_path = "config/config.yaml"
    configs = load_yaml(config_path)
    
    #edit config
    threads = configs['run']['threads'] 
    dataset = configs['run']['dataset']
    data_type = configs['run']['data_split']  
    
    model_path = configs['classification'][dataset]['model_path']
    tokenizer_dir = configs['classification'][dataset]['tokenizer_dir']

    model, tokenizer, device = load_model_classification(dataset, model_path, tokenizer_dir)
    
    settings = configs['type_run']   
    settings["dataset"] = f"{dataset}_"
    settings["wiki_cache"] = wiki_cache_data
    raw_data_file = configs['dataset'][dataset][data_type]
    raw_data = load_json(raw_data_file)
        
    if dataset == "medhop":
        settings['chrunk_mapping'] = load_json(configs['dataset'][dataset]['chrunk_mapping'])
        settings['normalize'] = True
        
    load_model(dataset, configs)
       
    for idx, (ID, sample) in enumerate(tqdm(raw_data.items(), desc="Processing")):
        if not (threads[0] <= idx < threads[1]):
            continue 
        question = sample['Question']
        
        # if sample['QIDX'] not in list_id:
        #     continue
        
        type_q, probabilities = classify_question(dataset, model, tokenizer, device, question)
        sample['type_q'] = type_q
        sample['probs'] = convert_ndarray(probabilities)
        
        if type_q in ['direct', 'parallel', 'comparison']:
            is_seq = False
        else:
            is_seq = True

        preprocess_sample(sample, settings, is_seq=is_seq, dataset=dataset, type_q=type_q, using_target_type=False)
        
        