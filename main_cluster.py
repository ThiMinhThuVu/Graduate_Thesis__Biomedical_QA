import os
import spacy
from tqdm import tqdm
from src.decomposition import decompose_question_parallel, decompose_question_sequential
from src.retrieval import retrieve_evidence, remove_retrieve_evidence, remove_entity_evidence, remove_specific_references
from src.extract_ner import extract_entities_from_evidence
from src.generate_answer import generate_combine_parallel_answers, _generate_answer
from src.candidate_cluster import process_get_final_answer
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
global embedding_model_cluster
extractor_model = None

global cache_ner_filter
cache_ner_filter = {}
mapping_data_answer = load_json('data/mapping_answer_gol.json')


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
        
        # Biomedical embedding
        # embedding_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        # embedding_model = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-base")  
        # embedding_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
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

def preprocess_sample(list_ner_q, sample, settings, is_seq=False, dataset=None, type_q=None,
            using_target_type = False, using_gen_old=False, has_extract_ner=False, using_decomp_new=False,
            target_ans_type=None, question_type_to_ner=None, using_prompt_yn=False):
    
    sample_id = str(sample['QIDX']) 
    print("[ID]: ", sample_id)
    paths = setup_paths(settings, sample_id)
    file_all_ans = paths['final_answer']
    cache_ner_filter[sample_id] = {}
    
    if os.path.exists(file_all_ans):
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
 
    for q_i, question in enumerate(list_qi, start=1):
        qi_number = f"/{q_i}"
        q_i = str(q_i)
        paths = setup_paths(settings, sample_id, qi_number)

        if q_i not in dict_ans:
            dict_ans[q_i] = {
                "final_qi": {
                    "question": "",
                    "short_answer": "",
                    "long_answer": "",
                }
            }
    
        path_output, n_path = decompose_question_sequential(question, sample_id, paths['decomp'], is_seq=is_seq, 
                            dataset=dataset, type_q=type_q, using_decomp_new=using_decomp_new, model="gpt-4o-mini")
        
        for path_i in range(n_path):
            path_i = str(path_i)
            subq_output = path_output[path_i]
            
            if path_i not in dict_ans[q_i]:
                dict_ans[q_i][path_i] = {}
             
                short_ans = ""
                long_ans = ""
                for step, item in subq_output.items():
                    step = str(step)
                    subquestion = item['subquestion']
                    search_kw_subq = item['search_keyword'] 
                    search_kw_subq_retrieval = search_kw_subq + f" {short_ans}"
                    if short_ans != "":
                        if short_ans.lower() not in list_ner_q:
                            list_ner_q.append(short_ans.lower())
                
                    if dataset == "wiki":
                        focus_spacy_based_entity_label = item.get('focus_spacy_based_entity_label', None)
                        target_ans_type = item.get('target_answer_type', None)
                    
                    if step not in dict_ans[q_i][path_i]:
                        dict_ans[q_i][path_i][step] = {} 

                    
                    if has_extract_ner:  
                        extracted_entities = extract_entities_from_evidence(extractor_model, extractor_model_2, extractor_model_3, 
                                                    search_kw_subq_retrieval, focus_spacy_based_entity_label)                         
                    else:
                        extracted_entities = ""
                        
                    # kmean doc search - found crawl wiki|ner docs
                    cluster_docs_ctx = retrieve_evidence(list_ner_q, sample, search_kw_subq_retrieval, settings,
                                                                target_ans_type, embedding_model, embedding_model_cluster, hash_id_cache, hash_id_crawl_wiki_cache)  
                    cluster_output = {}
                    for cluster_label, sample_cluster in cluster_docs_ctx.items():
                        context = "\n".join(sample_cluster['docs']) + " " + "\n".join(sample_cluster['crawl_doc'])                                                             
                        genered_answer = _generate_answer(subquestion, search_kw_subq_retrieval, extracted_entities, context, dataset, type_q, target_ans_type,
                                                    using_target_type, using_gen_old, has_extract_ner, using_prompt_yn)
                    
                        curr_anss = genered_answer['short_answer']
                        cluster_output[cluster_label] = {'llm':genered_answer,
                                                         'context': context
                        }
                        print(f"Short Answer for Step {step}-Cluster {cluster_label}: {curr_anss}")
                        
                    # Get final answer
                    final_answer = process_get_final_answer(subquestion, target_ans_type, cluster_output, mapping_data_answer, embedding_model_cluster)
                    short_ans = final_answer['short_answer']
                    long_ans = final_answer['long_answer']
                    dict_ans[q_i][path_i][step]["question"] = subquestion
                    dict_ans[q_i][path_i][step]["search_keyword"] = search_kw_subq      
                    dict_ans[q_i][path_i][step]["final_answer"] = final_answer      
                    dict_ans[q_i][path_i][step]["cluster_genered_answer"] = cluster_output
                    dict_ans[q_i][path_i][step]["retrieval_information"] = cluster_docs_ctx
           

            dict_ans[q_i]['final_qi']["question"] = question
            dict_ans[q_i]['final_qi']["short_answer"] = short_ans
            dict_ans[q_i]['final_qi']["long_answer"] = long_ans
                
       
        all_context += f"{q_i}.{question} - {dict_ans[q_i]['final_qi']['short_answer']}, {dict_ans[q_i]['final_qi']['long_answer']}\n\n"
    

    if n_qi > 1:
        combine_parallel_ans = generate_combine_parallel_answers(original_question, all_context, dataset=dataset, model="o3-mini")
        dict_ans['final_answer'] = combine_parallel_ans
        save_json(dict_ans, file_all_ans)
        print(f"Final answer: {combine_parallel_ans['short_answer']}")
        print(f"\nSaved to {file_all_ans}")
    else:
        dict_ans['final_answer'] = {
            'long_answer': dict_ans["1"]['final_qi']['long_answer'],
            'short_answer': dict_ans["1"]['final_qi']['short_answer'],
        }
       
        save_json(dict_ans, file_all_ans)
        print(f"Final answer: {dict_ans['final_answer']['short_answer']}")
        print(f"\nSaved to {file_all_ans}")    

if __name__ == "__main__":
    config_path = "config/config.yaml"
    configs = load_yaml(config_path)
    
    #edit config
    threads = [20, 50] # configs['run']['threads'] 
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
            using_prompt_yn = True
            # continue
        else:
            using_prompt_yn = False
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
            target_ans_type=target_ans_type, question_type_to_ner=question_type_to_ner, using_prompt_yn=using_prompt_yn)
    print(cnt)
    