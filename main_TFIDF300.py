import os
import spacy
from tqdm import tqdm
from src.decomposition import decompose_question_parallel, decompose_question_sequential
from src.retrieval import retrieve_evidence, remove_retrieve_evidence, remove_entity_evidence, remove_specific_references
from src.extract_ner import extract_entities_from_evidence
from src.generate_answer import generate_combine_parallel_answers, _generate_answer
from utils.util import *
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from transformers import AutoModel
import shutil

global hash_id_cache
global hash_id_crawl_wiki_cache
global extractor_model
global extractor_model_2 
global extractor_model_3
global embedding_model
extractor_model = None

global cache_ner_filter
cache_ner_filter = {}


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

    print(configs['cache'])
    hash_id_cache = load_json(configs['type_run']['cache']['hash_id']['retrieval'])
    hash_id_crawl_wiki_cache = load_json(configs['type_run']['cache']['hash_id']['crawl_wiki'])
    
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

def preprocess_sample(list_ner_q, sample, settings, is_seq=False, dataset=None, type_q=None,
            using_target_type = False, using_gen_old=False, has_extract_ner=False, using_decomp_new=False,
            target_ans_type=None, question_type_to_ner=None, model_mcq=None, mcq_add_no_ans=False, using_prompt_yn=False):
    
    sample_id = str(sample['QIDX']) 
    print("[ID]: ", sample_id)
    paths = setup_paths(settings, sample_id)
    file_all_ans = paths['final_answer']
    cache_ner_filter[sample_id] = {}
    
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
    
    if has_extract_ner: # Only extract snippet
        # focus_spacy_based_entity_label = ['GENE_OR_GENE_PRODUCT'] #["DISEASE", "CHEMICAL"]
        if target_ans_type in question_type_to_ner:
            focus_spacy_based_entity_label = question_type_to_ner[target_ans_type]                                
        else: #Other
            focus_spacy_based_entity_label = ["DISEASE", "CHEMICAL"]                              

    if os.path.exists(file_all_ans):
        dict_ans = load_json(file_all_ans)
    else:
        dict_ans = {
            "QIDX": sample_id,
            "original_question": original_question,
            "type_q": sample['type_q'],
            "focus_spacy_based_entity_label": focus_spacy_based_entity_label,
            "target_answer_type": target_ans_type,
            "key_entities": list_ner_q
        }

    all_context = ""
    if target_ans_type == 'correctness':
        settings['max_candidates'] = 1
        
        
    for q_i, question in enumerate(list_qi, start=1):
        qi_number = f"/{q_i}"
        q_i = str(q_i)
        paths = setup_paths(settings, sample_id, qi_number)

        if q_i not in dict_ans:
            dict_ans[q_i] = {
                "final_qi": {
                    "question": "",
                    "search_keyword": "",
                    "short_answer": "",
                    "long_answer": "",
                    "retrieval_information": "",
                    "prefix": "",
                    "name": "",
                    "suffix": ""
                }
            }
    
        path_output, n_path = decompose_question_sequential(question, sample_id, paths['decomp'], is_seq=is_seq, 
                            dataset=dataset, type_q=type_q, using_decomp_new=using_decomp_new, model="gpt-4o-mini")
        
        for path_i in range(n_path):
            path_i = str(path_i)
            subq_output = path_output[path_i]
            
            if path_i not in dict_ans[q_i]:
                dict_ans[q_i][path_i] = {}
                # print(f"dict_ans[q_i][path_i]: {dict_ans[q_i][path_i]}")
             

            for module in [[True, False, False, False, False], [False, True, False, False, False], [False, False, True, False, False], 
                           [False, False, False, True, False], [False, False, False, False, True]]:
                settings['get_gg'] = module[0]
                settings['get_wiki'] = module[1]    
                settings['get_crawl_wiki'] = module[2]
                settings['get_ner_wiki'] = module[3]
                settings['get_sparql_context'] = module[4]

                if settings['get_gg']:
                    source = 'google'
                elif settings['get_wiki']:
                    source = 'wiki'
                elif settings['get_crawl_wiki']:
                    source = 'crawl_wiki'
                elif settings['get_ner_wiki']:
                    source = 'crawl_ner_wiki'
                elif settings['get_sparql_context']:
                    source = 'crawl_sparql_wikidata'
                
                
                short_ans = ""
                
                source_have_run = False
                for step, item in subq_output.items():
                    step = str(step)
                    subquestion = item['subquestion']
                    search_kw_subq = item['search_keyword'] 
                    search_kw_subq_retrieval = search_kw_subq + f" {short_ans}"
                    is_duplicate = False
                    if step == "0":
                        key_entities = item['key_entity']   
                    else:
                        key_entities = [short_ans]
                        if short_ans.lower() not in list_ner_q:
                            list_ner_q.append(short_ans.lower())
                
                    if dataset == "wiki":
                        focus_spacy_based_entity_label = item.get('focus_spacy_based_entity_label', None)
                        target_ans_type = item.get('target_answer_type', None)
                    
                    
                    if step not in dict_ans[q_i][path_i]:
                        dict_ans[q_i][path_i][step] = {} 


                    if "source_output" not in dict_ans[q_i][path_i][step]:
                        dict_ans[q_i][path_i][step]["source_output"] = {}
                        
                    if source not in dict_ans[q_i][path_i][step]["source_output"]:
                        dict_ans[q_i][path_i][step]["source_output"][source] = {}
                        # Multiple Choice
                        candidates = []
                        idx_del = 0
                        current_short = short_ans
                        if has_extract_ner: # Only extract snippet        
                            all_snippet, current_retrieval = retrieve_evidence(list_ner_q, sample, key_entities, search_kw_subq_retrieval, settings,
                                                                    target_ans_type, embedding_model, hash_id_cache, hash_id_crawl_wiki_cache)                  
                            information_to_extract = search_kw_subq_retrieval + " " + all_snippet
                            extracted_entities = extract_entities_from_evidence(extractor_model, extractor_model_2, extractor_model_3, 
                                                        information_to_extract, focus_spacy_based_entity_label)                         
                        else:
                            extracted_entities = ""
                        # Clean current_retrieval using remove_specific_references if possible (no candidates yet, so no long_answer)
                        # This is only relevant after the first candidate is generated, so skip here
                    else:
                        src_candidates = dict_ans[q_i][path_i][step]["source_output"][source]["candidate"]
                        if  len(src_candidates) >= settings['max_candidates']:
                            source_have_run = True
                            continue
                        else:
                            # Multiple Choice
                            candidates = src_candidates #lấy lại số choice hiện có
                            idx_del = len(candidates)                            
                            current_short = candidates[-1]['short_answer'].lower()
                            current_retrieval = candidates[-1]['retrieval_information'] 
                            extracted_entities = candidates[-1]['extracted_entities']     
                            # Nếu đã có candidate trước đó, dùng long_answer của candidate cuối để làm sạch current_retrieval
                            last_long_answer = candidates[-1].get('long_answer', '') if len(candidates) > 0 else ''
                            if last_long_answer and current_retrieval:
                                cleaned_retrieval = remove_specific_references(last_long_answer, current_retrieval)
                                current_retrieval = cleaned_retrieval
                   
                    print(f"\nSource [{source}] Processing Step {step}: {subquestion} ---")
                                                
                    while idx_del < settings['max_candidates']:
                        idx_del += 1
                        
                        if idx_del > 1: # trong list n cand check [0,n-2]
                            is_duplicate = any(c['short_answer'].lower().strip() == current_short.lower().strip() for c in candidates[:-1])
                        if is_duplicate and len(candidates) > 0:
                            candidates.append(candidates[-1])
                            continue
                        
                        if idx_del > 1 and (current_short != "" and current_short.lower() != "insufficient information"):
                            # Use long_answer to remove specific references from retrieval_information
                            long_answer = normalized_answer["long_answer"] if "long_answer" in normalized_answer else ""
                            new_retrieval = remove_specific_references(long_answer, current_retrieval)
                            extracted_entities = remove_entity_evidence(extracted_entities, current_short)
                            if new_retrieval == current_retrieval:
                                break
                            else:
                                current_retrieval = new_retrieval
                        elif len(candidates) > 0: #giữ nguyên kq của choice trước nó
                            candidates.append(candidates[-1])
                            continue                       
                                                                 
                        normalized_answer = _generate_answer(subquestion, search_kw_subq_retrieval, extracted_entities, current_retrieval, dataset, type_q, target_ans_type,
                                                        using_target_type, using_gen_old, has_extract_ner, using_prompt_yn)
                        current_short = normalized_answer['short_answer']

                        print(f"[MCQ] short ans lần {idx_del}: ", current_short)                      
                        
                        if current_retrieval == "":
                            candidates.append({
                                "short_answer": "insufficient information",
                                "long_answer": "No context",
                                "retrieval_information": current_retrieval,
                                "extracted_entities": extracted_entities,
                                "full_data": {
                                    "short_answer": "insufficient information",
                                    "long_answer": "No context"
                                }
                            })                                                            
                        else:
                            candidates.append({
                                "short_answer": current_short,
                                "long_answer": normalized_answer["long_answer"],
                                "retrieval_information": current_retrieval,
                                "extracted_entities": extracted_entities,
                                "full_data": normalized_answer # Lưu lại để dùng sau khi chọn
                            })
                        
                        
                        if current_short.lower() in ['yes', 'no']:
                            break
                        
                        
                    if mcq_add_no_ans:
                        normalized_answer_no_ans = {
                            'short_answer': "No answer",
                            'long_answer': "The remaining answer options are not appropriate."
                        }
                        candidates.append({
                            "short_answer": "No answer",
                            "long_answer": normalized_answer_no_ans["long_answer"],
                            "retrieval_information": "",
                            "extracted_entities": extracted_entities,
                            "full_data": normalized_answer_no_ans # Lưu lại để dùng sau khi chọn
                        })
                  
                
                    print(f"Short Answer for Step {step}: {short_ans}")

                    dict_ans[q_i][path_i][step]["question"] = subquestion
                    dict_ans[q_i][path_i][step]["search_keyword"] = search_kw_subq            
                    # dict_ans[q_i][path_i][step]["source_output"][source]["short_answer"] = short_ans
                    # dict_ans[q_i][path_i][step]["source_output"][source]["long_answer"] = long_ans
                    # dict_ans[q_i][path_i][step]["source_output"][source]["retrieval_information"] = retrieval_information
                    dict_ans[q_i][path_i][step]["source_output"][source]["candidate"] = candidates

                    # if type_q == "inference":
                    #     dict_ans[q_i][path_i][step]['prefix'] = prefix_ans
                    #     dict_ans[q_i][path_i][step]['name'] = name_ans
                    #     dict_ans[q_i][path_i][step]['suffix'] = suffix_ans

                # continue
                save_json(dict_ans, file_all_ans)
                print("SAVE file: ", file_all_ans)

                if source_have_run:
                    continue
                # save_json(dict_ans, file_all_ans)             

                dict_ans[q_i]['final_qi']["question"] = question
                dict_ans[q_i]['final_qi']["search_keyword"] = search_kw_subq
                # dict_ans[q_i]['final_qi']["short_answer"] = short_ans
                # dict_ans[q_i]['final_qi']["long_answer"] = long_ans
                dict_ans[q_i]['final_qi']["candidate"] = []# qi_candidates
                # dict_ans[q_i]['final_qi']["retrieval_information"] = retrieval_information
                   
                # if type_q == "inference":
                #     dict_ans[q_i]['final_qi']['prefix'] = prefix_ans
                #     dict_ans[q_i]['final_qi']['name'] = name_ans
                #     dict_ans[q_i]['final_qi']['suffix'] = suffix_ans   

                        # For 'inference' questions, the first valid path is sufficient.
                        # break
                        
        # if cache_source_have_run == select_source_mcq:
        #     continue
            
        #chỉ dùng khi là sequence, nên ksao
        # all_context += f"{q_i}.{question} - {dict_ans[q_i]['final_qi']['short_answer']}, {dict_ans[q_i]['final_qi']['long_answer']}\n\n"
    
    save_json(dict_ans, file_all_ans)
    print("SAVE file: ", file_all_ans)
    return 

    if n_qi > 1:
        combine_parallel_ans = generate_combine_parallel_answers(original_question, all_context, dataset=dataset, model="o3-mini")
        dict_ans['final_answer'] = combine_parallel_ans
        dict_ans['final_answer']['retrieval_information'] = all_context
        save_json(dict_ans, file_all_ans)
        print(f"Final answer: {combine_parallel_ans['short_answer']}")
        print(f"\nSaved to {file_all_ans}")
    else:
        dict_ans['final_answer'] = {
            'long_answer': dict_ans["1"]['final_qi']['long_answer'],
            'short_answer': dict_ans["1"]['final_qi']['short_answer'],
            'retrieval_information': dict_ans["1"]['final_qi']['retrieval_information']
        }
        if type_q == "inference":
            dict_ans['final_answer']['prefix'] = dict_ans["1"]['final_qi']['prefix']
            dict_ans['final_answer']['name'] = dict_ans["1"]['final_qi']['name']
            dict_ans['final_answer']['suffix'] = dict_ans["1"]['final_qi']['suffix']
            
        save_json(dict_ans, file_all_ans)
        print(f"Final answer: {dict_ans['final_answer']['short_answer']}")
        print(f"\nSaved to {file_all_ans}")    

if __name__ == "__main__":
    config_path = "config/config.yaml"
    configs = load_yaml(config_path)
    
    #edit config
    threads = [0, 250] # configs['run']['threads'] 
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
    reset_directory(r"data/cache/filter_ner_wikipedia")
    reset_directory(r"data/cache/filter_wikipedia")
    load_model(dataset, configs, using_embedding_model=False) #False = tfidf, true = embedding_model
    # ner_version = 4
    ner_version = "#3"
    metadata = load_json('data/metadata_medhop.json')
    model_mcq_name = "o3-mini" #"gpt-4.1-mini" #o3-mini
    mcq_add_no_ans = False # True #False
    using_prompt_yn = False # False
    meta_ner = load_json(r'data/metadata_entity.json')
    cnt = 0
    for idx, (ID, sample) in enumerate(tqdm(raw_data.items(), desc="Processing")):
        if not (threads[0] <= idx < threads[1]):
            continue 
        
        qidx = sample['QIDX']
        question = sample['Question']
        target_ans_type = metadata[qidx]['target_answer_type']
        
        # CONFIG YESNO
        check_yn = target_ans_type == 'correctness'
        if check_yn:
            continue
        
        type_q = metadata[qidx]['type_q']    
        
        #CONFIG    
        if type_q != "direct":
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
        
        preprocess_sample(list_ner_q, sample, settings, is_seq=is_seq, dataset=dataset, type_q=type_q, 
            using_target_type=False, using_gen_old=True, has_extract_ner=True, using_decomp_new=False,
            target_ans_type=target_ans_type, question_type_to_ner=question_type_to_ner,model_mcq=model_mcq_name,
            mcq_add_no_ans=mcq_add_no_ans, using_prompt_yn=using_prompt_yn)
    print(cnt)
    