import os
import spacy
import pickle
from tqdm import tqdm
from src.decomposition import *
from src.retrieval import *
from src.extract_ner import *
from src.generate_answer import *
from src.all_run import *
from utils.norm_wikidata_pair import *
from utils.util import *

import pandas as pd

global wiki_cache_data
global nlp_bio
global nlp_standard
wiki_cache_data = None
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


def generate_answer_for_subq(subquestion, key_entities, retrieval_information, step, settings, paths, dataset=None, type_q=None):
    if dataset in ["wiki", "cwq"]:
        return generate_answer(subquestion, key_entities, retrieval_information, dataset=dataset, type_q=type_q)
    else:
        ner_filepath = paths['ner']
        output_ner = load_json(ner_filepath) if os.path.exists(ner_filepath) else {}
        
        if step in output_ner and settings['run'] != "R7":
            bio_entities = output_ner[step]
        else:
            print(f"Extracting entities.")
            query_for_ner = subquestion + " " + retrieval_information
            bio_entities = extract_with_models(query_for_ner, nlp_bio, nlp_standard, dataset=dataset)
            output_ner[step] = bio_entities
            save_json(output_ner, ner_filepath)
            key_entities += " BioEntities: " + ", ".join(bio_entities)

        return generate_answer(subquestion, key_entities, retrieval_information, dataset=dataset, type_q=type_q)

def preprocess_sample(sample, settings, is_seq=False, dataset=None, type_q=None, query_wikidata=None):
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
                # if len(subq_output.keys()) != len(qi_final_answers[path_i].keys()):
                #     print("Path này không tìm được answer, bỏ qua")
                #     dict_ans[q_i][path_i]['0'] = qi_final_answers[path_i]['0']
                #     dict_ans[q_i][path_i]['0']['question'] = subq_output['0']['subquestion']
                #     dict_ans[q_i][path_i]['0']['search_keyword'] = subq_output['0']['search_keyword']
                #     dict_ans[q_i][path_i]['0']['key_entity'] = subq_output['0']['key_entity']
                #     continue
                
                short_ans = ""
                long_ans = ""
                for step, item in subq_output.items():
                    step = str(step)   
                    subquestion = item['subquestion']
                    search_kw_subq = item['search_keyword']
                    if settings['search_key_entity'] == "True": 
                        if item['key_entity'] == "" or item['key_entity'] == None: 
                            key_entities = short_ans
                        else:
                            key_entities = item['key_entity']        
                    else:
                        key_entities = search_kw_subq + " " + short_ans
                    
                    retrieval_information = perform_retrieval(sample, key_entities, search_kw_subq, step, paths, settings)

                    if step == '0':
                        short_ans = qi_final_answers[path_i][step]["short_answer"]
                        long_ans = qi_final_answers[path_i][step]["long_answer"]     
                        last_key_entity = key_entities                  
                        last_retrieval_information = retrieval_information
                        q_relation_child = search_kw_subq
                    if step == '1':
                        if type_q in ["inference", "compositional"] and dataset == "wiki" and query_wikidata==True: 
                            q_relation_parent = search_kw_subq
                            answer_generation = query_hop2_with_wikidata(type_q, original_question, key_entities, last_key_entity, q_relation_parent, q_relation_child, retrieval_information, last_retrieval_information)
                            normalized_v2 = answer_generation
                            short_ans = answer_generation["short_answer"]
                            long_ans = answer_generation["long_answer"]
                            qi_final_answers[path_i][step] = answer_generation
                        else:
                            if step not in qi_final_answers[path_i]:
                                continue
                            short_ans = qi_final_answers[path_i][step]["short_answer"]
                            long_ans = qi_final_answers[path_i][step]["long_answer"]

                        if type_q == "inference" and dataset == "wiki":
                            prefix_ans = ""
                            name_ans = ""
                            suffix_ans = ""
                            
                    if step not in dict_ans[q_i][path_i]:
                        dict_ans[q_i][path_i][step] = {}
                        
                    dict_ans[q_i][path_i][step]["question"] = subquestion
                    dict_ans[q_i][path_i][step]["search_keyword"] = search_kw_subq
                    dict_ans[q_i][path_i][step]["key_entity"] = key_entities
                    dict_ans[q_i][path_i][step]["short_answer"] = short_ans
                    dict_ans[q_i][path_i][step]["long_answer"] = long_ans
                    dict_ans[q_i][path_i][step]["retrieval_information"] = retrieval_information

                    if type_q in ["inference", "compositional"] and dataset == "wiki" and query_wikidata==True:
                        if step == '1':
                            dict_ans[q_i][path_i][step]['normalized_v2'] = normalized_v2
                    # else:
                    #     dict_ans[q_i][path_i][step]['prefix'] = prefix_ans
                    #     dict_ans[q_i][path_i][step]['name'] = name_ans
                    #     dict_ans[q_i][path_i][step]['suffix'] = suffix_ans
                        
            else:
                short_ans = ""
                qi_final_answers[path_i] = {}
                
                for step, item in subq_output.items():
                    step = str(step)
                    subquestion = item['subquestion']
                    search_kw_subq = item['search_keyword']
                    
                    if settings['search_key_entity'] == "True": 
                        if item['key_entity'] == "" or item['key_entity'] == None: 
                            key_entities = short_ans
                        else:
                            key_entities = item['key_entity']        
                    else:
                        key_entities = search_kw_subq + " " + short_ans
                
                    print(f"\n--- Processing Step {step}: {subquestion} ---")
                    retrieval_information = perform_retrieval(sample, key_entities, search_kw_subq, step, paths, settings)
                    # print(f'Retrieval info: {retrieval_information}')

                    if query_wikidata:
                        if step == '0':
                            answer_generation = generate_answer_for_subq(subquestion, key_entities, retrieval_information, step, paths, settings, dataset=dataset, type_q=type_q)
                            short_ans = answer_generation["short_answer"]
                            long_ans = answer_generation["long_answer"]
                        
                            if type_q == "inference" and dataset == "wiki":
                                prefix_ans = answer_generation['prefix']
                                name_ans = answer_generation['name']
                                suffix_ans = answer_generation['suffix']
                            
                            if dataset == "wiki": 
                                q_relation_child = search_kw_subq
                                last_key_entity = key_entities
                                last_retrieval_information = retrieval_information
                                
                        elif step == '1':
                            if dataset == "wiki": 
                                q_relation_parent = search_kw_subq
                                answer_generation = query_hop2_with_wikidata(type_q, original_question, key_entities, last_key_entity, q_relation_parent, q_relation_child, retrieval_information, last_retrieval_information)     
                                normalized_v2 = answer_generation
                            else:
                                answer_generation = generate_answer_for_subq(subquestion, key_entities, retrieval_information, step, paths, settings, dataset=dataset, type_q=type_q)
                            
                            short_ans = answer_generation["short_answer"]
                            long_ans = answer_generation["long_answer"]
                        
                            if type_q == "inference" and dataset == "wiki":
                                prefix_ans = ""
                                name_ans = ""
                                suffix_ans = ""
                    else:
                        retrieval_information = perform_retrieval(sample, key_entities, search_kw_subq, step, paths, settings)
                    
                        answer_generation = generate_answer_for_subq(subquestion, key_entities, retrieval_information, step, paths, settings, dataset=dataset, type_q=type_q)
                        short_ans = answer_generation["short_answer"]
                        long_ans = answer_generation["long_answer"]
                        
                        if type_q == "inference" and dataset == "wiki":
                            prefix_ans = answer_generation['prefix']
                            name_ans = answer_generation['name']
                            suffix_ans = answer_generation['suffix']

                        qi_final_answers[path_i][step] = answer_generation
                    
                    print(f"Short Answer for Step {step}: {short_ans}")

                    dict_ans[q_i][path_i][step] = {} 
                    dict_ans[q_i][path_i][step]["question"] = subquestion
                    dict_ans[q_i][path_i][step]["search_keyword"] = search_kw_subq
                    dict_ans[q_i][path_i][step]["key_entity"] = key_entities
                    dict_ans[q_i][path_i][step]["short_answer"] = short_ans
                    dict_ans[q_i][path_i][step]["long_answer"] = long_ans
                    dict_ans[q_i][path_i][step]["retrieval_information"] = retrieval_information
                    if type_q == "inference" and dataset == "wiki":
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
                dict_ans[q_i]['final_qi']["short_answer"] = short_ans
                dict_ans[q_i]['final_qi']["long_answer"] = long_ans
                dict_ans[q_i]['final_qi']["retrieval_information"] = retrieval_information

                if type_q in ["inference", "compositional"] and dataset == "wiki" and query_wikidata==True:
                    dict_ans[q_i]['final_qi']['normalized_v2'] = normalized_v2
                # else:
                #     dict_ans[q_i]['final_qi']['prefix'] = prefix_ans
                #     dict_ans[q_i]['final_qi']['name'] = name_ans
                #     dict_ans[q_i]['final_qi']['suffix'] = suffix_ans   

                # For 'inference' questions, the first valid path is sufficient.
                if type_q == "inference":
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
        if type_q in ["inference", "compositional"] and dataset == "wiki" and query_wikidata==True:
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
    config_path = "config.json"
    configs = load_json(config_path)
    datasets = ["medhop", "bioasq", "cwq", "wiki"]
    batchs = ["b1", "b2", "b3", "b4"] # medhop chỉ có 2 batch (unconfused, confused)
    runs = ["R1", "R4", "R5", "R6", "R7", "R10", "R11", "R12", "R13", "R14"]
    list_type = ["bridge_comparison", "comparison", "compositional", "inference"]
    type_dataset = ['dev', 'test']
    
    #edit config
    query_hop2_use_wikidata = False #True #False
    run = runs[9]  #1: R4, 4: R7,  6: R11, 7: R12, 8: R13, 9: R14
    threads = [0, 1] # Điền threads vào đây: ví dụ [0, 100]
    dataset = datasets[3] # Chọn dataset: medhop, bioasq, cwq, wiki
    batch = batchs[0]    
    important = True # False or True: Chỉ dùng cho medhop
    type_ = list_type[0] # Chọn loại câu hỏi: bridge_comparison, comparison, compositional, inference
    data_type = type_dataset[0]  # dev or test
    
    settings = configs[dataset][run]
    settings["wiki_cache"] = wiki_cache_data
    raw_data_file = configs[dataset]["raw_data_file"][batch]
    
        
    if dataset == "medhop":
        load_spacy_model()
        important_file = configs[dataset]["important_file"]
        chrunk_mapping_file = configs[dataset]["chrunk_mapping_file"]
        chrunk_mapping = load_json(chrunk_mapping_file)
        settings["chrunk_mapping"] = chrunk_mapping 
        
        if important:
            settings['dataset'] = "medhop_important_"
            raw_data = load_json(important_file)
            for idx, (ID, sample) in enumerate(tqdm(raw_data.items(), desc="Processing")):
                if not (threads[0] <= idx < threads[1]):
                    continue 
                
                is_seq = sample["is_sequence"]
                sample['type'] = ''
                
                if run == "R5":
                    is_seq = False                
                if run == "R7":
                    preprocess_sample_R7(sample, settings, dataset=dataset)
                else:
                    preprocess_sample(sample, settings, is_seq=is_seq, dataset=dataset, type_q='')
        else:
            if batch == "b1":
                settings['dataset'] = "medhop_unconfused_"
            elif batch == "b2":
                settings['dataset'] = "medhop_confused_"
            raw_data = pd.read_csv(raw_data_file)
                
            for index, sample in raw_data.iloc[threads[0]:threads[1]].iterrows():       
                is_seq = sample["is_sequence"]     
                sample['type'] = ''
                           
                if run == "R5":
                    is_seq = False
                if run == "R7":
                    preprocess_sample_R7(sample, settings, dataset=dataset)
                else:
                    preprocess_sample(sample, settings, is_seq=is_seq, dataset=dataset, type_q='')
                    
    elif dataset == "wiki":   
        settings['dataset'] = f"wiki_{type_}_"
        raw_data_file = f"data/wiki/{data_type}/{data_type}_{type_}.json"
        raw_data = load_json(raw_data_file)
        
        list_ID = [   
            "dfe496880bdd11eba7f7acde48001122",
            "0051fad20bde11eba7f7acde48001122",
            "6ebdbede0bde11eba7f7acde48001122",
            "0189caf20bdd11eba7f7acde48001122"
        ]
                
        for idx, (ID, sample) in enumerate(tqdm(raw_data.items(), desc="Processing")):
            if not (threads[0] <= idx < threads[1]):
                continue 
            
            # if ID not in list_ID:
            #     continue

            type_q = sample['type']
            is_seq = sample["is_sequence"]
            
            if run == "R5":
                is_seq = False
            if run == "R7":
                preprocess_sample_R7(sample, settings, dataset=dataset)
            if run == "R12":
                preprocess_sample_R12(sample, settings, dataset=dataset, type_q=type_q)
            else:
                preprocess_sample(sample, settings, is_seq=is_seq, dataset=dataset, type_q=type_q, query_wikidata=query_hop2_use_wikidata)
