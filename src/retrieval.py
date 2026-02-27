from src.gpt_request import search_google
from utils.crawl_wikipage import crawl_wikipedia_page
from utils.get_abs_wiki_history import get_abs_wiki_history
from utils.process_wikidata_api import process_fillter_wikidata_api, match_relation_id
from utils.util import *
from utils.hash import register_query_cache, make_cache_filename, normalize_query
from utils.get_section_priority_crawl import *
from sklearn.cluster import KMeans
import numpy as np
        
import re
import os
import math

def retrieval_google(search_kw, filepath, max_results=20):
    if os.path.exists(filepath):
        output = load_json(filepath)
        if 'google' in output:
            result = output['google']['search_results']
            if result == []:
                print("[API SEARCH] HẾT LƯỢT SEARCHs")
                print("Search kw: ", search_kw)
                return None
            
            print(f"Cache hit Google Search. Loading from file.")
            return result
    else:
        output = {}
            
    max_retries = 3
    for attempt in range(max_retries):
        result = search_google(search_kw, max_results=max_results)
        if result != []:
            break
        else:
            print(f"[API SEARCH] HẾT LƯỢT SEARCHs (attempt {attempt+1}/{max_retries})")
            print("Search kw: ", search_kw)
            return None
          
    output['google'] = {
        'search_kw': search_kw,
        "search_results": result                
    }
  
    save_json(output, filepath)        
    return result

def retrieval_wiki(search_kw, filepath, max_results=10):
    if os.path.exists(filepath):
        output = load_json(filepath)
        if 'wiki' in output:
            print(f"Cache hit Wikipedia Search. Loading from file.")
            return output['wiki']['search_results']
    else:
        output = {}
     
    result = search_google(search_kw + " site:wikipedia.org", max_results=max_results)           
    
    output['wiki'] = {
        'search_kw': search_kw,
        "search_results": result                
    }
  
    save_json(output, filepath)        
    return result


def format_search_results(search_results, topk=5, search_type='GG'):
    context_parts = []
    for i, result in enumerate(search_results):            
        if i == topk:
            break
        
        context_parts.append(f"-Reference: [{search_type}_{i+1}]: {result['title']} {result['snippet']}")
                                       
    return "\n".join(context_parts), context_parts

def format_results_wiki_abs(search_results, filepath, topk=5, bf_ts=None):
    context_parts = []
    cnt = 0
    if os.path.exists(filepath):
        output = load_json(filepath)
    else:
        output = {}
        
    if 'wikipedia_abs' in output:
        for link, item in output['wikipedia_abs'].items():
            if cnt == topk:
                break
            if "List_"  in link:
                continue                
            context_parts.append(f"- {item['title']}: {item['abstract']}")
            cnt += 1
            
        return "\n".join(context_parts)
    else:       
        output['wikipedia_abs'] = {}
            
    for i, result in enumerate(search_results):            
        if i == topk:
            break
        title = result['title']
        url = result['link']
        if "List_" in url:
            continue

        abs_wiki = get_abs_wiki_history(title, bf_ts)
        context_parts.append(f"- {title}: {abs_wiki}")
        
        output["wikipedia_abs"][result['link']] = {'title': title,
                                                    'snippet': result['snippet'],
                                                    'abstract': abs_wiki}
    save_json(output, filepath)
    return "\n".join(context_parts)

def get_paragraphs_wiki_result(link, sample, subquestion, target_answer_type=None, embedding_model=None, max_docs_token=300, percentage=None):    
    list_para_sim = {}
    list_para_text = []
    index_doc = 0
    key_section = sample['sections'].keys()
    key_section = [section for section in key_section if section.lower() not in move_section_list]
    
    for heading, item in sample['sections'].items():
        if heading.lower() in move_section_list:
            continue
        
        for idx, p in enumerate(item['content']):
            list_para_sim[index_doc] = {
                "link": link,
                "idx_sec": idx,
                "heading": heading,
                "text": p
            }
            index_doc += 1
            list_para_text.append(p)
    
    if len(list_para_text) == 0:
        print("[crawl wiki] Link này rỗng")
        return None
                
    if embedding_model:
        similarity = find_most_relevant_doc_embedding(subquestion, list_para_text, embedding_model)
    else:
        similarity = find_most_relevant_doc(subquestion, list_para_text)
    
    #THUVTM
    # Tính tổng số token của toàn bộ doc
    if percentage:
        total_tokens = sum(len(p.split()) for p in list_para_text)
        max_token = int(total_tokens * percentage)
    else:
        max_token = max_docs_token
         
    count_token = 0
    selected_idx_sentences = []
    
    for item in similarity:
        if count_token >= max_token: #max_docs_token
            break
        
        score = item["score"]
        idx_doc = item["index"]
        para = list_para_sim[idx_doc]["text"]
        selected_idx_sentences.append({
            "idx_doc": idx_doc,
            "score": score,
            "text": para,
        })
        count_token += len(para.split())
        
        # THêm hàng xóm
        index_section = list_para_sim[idx_doc]["idx_sec"]
        selected_idx_set = {s["idx_doc"] for s in selected_idx_sentences}
        
        if index_section > 0:
            idx_nbr = idx_doc-1 #idx doc
            if idx_nbr not in selected_idx_set:
                nbr_para = list_para_sim[idx_nbr]["text"]
                selected_idx_sentences.append({
                    "idx_doc": idx_nbr,#idx của doc
                    "score": score,
                    "text": nbr_para,
                })
                count_token += len(nbr_para.split())
        
        if index_section < len(sample['sections'][list_para_sim[idx_doc]["heading"]]['content']) - 1:
            idx_nbr = idx_doc + 1 #doc
            if idx_nbr not in selected_idx_set:
                nbr_para = list_para_sim[idx_nbr]["text"]
                selected_idx_sentences.append({
                    "idx_doc": idx_nbr,#idx của doc
                    "score": score,
                    "text": nbr_para,
                })
                count_token += len(nbr_para.split())
                
    list_idx = [s["idx_doc"] for s in selected_idx_sentences]                 
    # thêm li
    last_section = key_section[-1]
    cnt_li = 0
    list_idx_last_sect = []
    
    for idx_doc, item in list_para_sim.items():
        if idx_doc not in list_idx and item["text"].strip().startswith("<info>"):
            heading = item["heading"]
            if heading.lower() in move_section_list:
                continue
            
            if heading == last_section:                    
                cnt_li += 1
                list_idx_last_sect.append(idx_doc)
                    
            selected_idx_sentences.append({
                "idx_doc": idx_doc,#doc
                "score": None,  
                "text": item["text"],
            })
    
    # Nếu section cuối có > 30 li thì xóa các li của section cuối
    if cnt_li > 30:
        selected_idx_sentences = [item for item in selected_idx_sentences if item["idx_doc"] not in list_idx_last_sect]
        
    selected_idx_sentences = sorted(selected_idx_sentences, key=lambda x: x["idx_doc"])      

    res_paragraphs = ""
    res_output = {}
    
    # xếp theo thứ tự của doc, section
    for head in key_section:
        text_sec = ""
        for item in selected_idx_sentences:
            idx_doc = item["idx_doc"]#doc
            text = item["text"]
            heading = list_para_sim[idx_doc]["heading"]
            if head == heading:
                if item["text"].strip().startswith("<info>"):
                    text = text.replace("<info>", "", 1).strip()
                text_sec += f"{text.strip()} "
        if text_sec != "":
            res_output[head] = text_sec  

    # THUVTM: DEFAULT: NẾU target_answer_type thì section thuộc wanted_sections sẽ luôn được thay thế bằng content gốc
    if target_answer_type:        
        wanted_sections = get_sections_for_target_type(target_answer_type)
        if 'introduction' not in wanted_sections:
            wanted_sections.append('introduction')
            
        for sec_name in wanted_sections:
            for head in key_section:
                if normalize_section_name(head) == sec_name:
                    section_content = sample['sections'][head]['content']
                    text_sec = " ".join([
                        t.replace("<info>", "", 1).strip() if t.strip().startswith("<info>") else t.strip()
                        for t in section_content
                    ])
                    # Luôn thay thế, kể cả đã có context trước đó
                    res_output[head] = text_sec

    for head, sec_text in res_output.items():
        res_paragraphs += f"Section: {head}\n"
        res_paragraphs += f"{sec_text.strip()}\n"    
    
    return res_paragraphs



def get_paragraphs_wiki_result_with_sec(link, sample, subquestion, target_answer_type=None, embedding_model=None, max_docs_token=300, percentage=0.5):    
    list_para_sim = {}
    list_para_text = []
    index_doc = 0
    key_section = sample['sections'].keys()
    key_section = [section for section in key_section if section.lower() not in move_section_list]
    
    for heading, item in sample['sections'].items():
        if heading.lower() in move_section_list:
            continue
        
        for idx, p in enumerate(item['content']):
            list_para_sim[index_doc] = {
                "link": link,
                "idx_sec": idx,
                "heading": heading,
                "text": p
            }
            index_doc += 1
            list_para_text.append(p)
    
    if len(list_para_text) == 0:
        print("[crawl wiki] Link này rỗng")
        return None
                
    if embedding_model:
        similarity = find_most_relevant_doc_embedding(subquestion, list_para_text, embedding_model)
    else:
        similarity = find_most_relevant_doc(subquestion, list_para_text)
    
    #THUVTM
    # Tính tổng số token của toàn bộ doc
    if percentage:
        total_tokens = sum(len(p.split()) for p in list_para_text)
        max_token = int(total_tokens * percentage)
    else:
        max_token = max_docs_token
         
    count_token = 0
    selected_idx_sentences = []
    
    for item in similarity[:3]:        
        score = item["score"]
        idx_doc = item["index"]
        para = list_para_sim[idx_doc]["text"]
        selected_idx_sentences.append({
            "idx_doc": idx_doc,
            "score": score,
            "text": para,
        })
        count_token += len(para.split())
        
        # THêm hàng xóm
        index_section = list_para_sim[idx_doc]["idx_sec"]
        selected_idx_set = {s["idx_doc"] for s in selected_idx_sentences}
        
        if index_section > 0:
            idx_nbr = idx_doc-1 #idx doc
            if idx_nbr not in selected_idx_set:
                nbr_para = list_para_sim[idx_nbr]["text"]
                selected_idx_sentences.append({
                    "idx_doc": idx_nbr,#idx của doc
                    "score": score,
                    "text": nbr_para,
                })
                count_token += len(nbr_para.split())
        
        if index_section < len(sample['sections'][list_para_sim[idx_doc]["heading"]]['content']) - 1:
            idx_nbr = idx_doc + 1 #doc
            if idx_nbr not in selected_idx_set:
                nbr_para = list_para_sim[idx_nbr]["text"]
                selected_idx_sentences.append({
                    "idx_doc": idx_nbr,#idx của doc
                    "score": score,
                    "text": nbr_para,
                })
                count_token += len(nbr_para.split())
                
    selected_idx_sentences = sorted(selected_idx_sentences, key=lambda x: x["idx_doc"])      

    res_paragraphs = ""
    res_output = {}
    
    # xếp theo thứ tự của doc, section
    for head in key_section:
        text_sec = ""
        for item in selected_idx_sentences:
            idx_doc = item["idx_doc"]#doc
            text = item["text"]
            heading = list_para_sim[idx_doc]["heading"]
            if head == heading:
                if item["text"].strip().startswith("<info>"):
                    text = text.replace("<info>", "", 1).strip()
                text_sec += f"{text.strip()} "
        if text_sec != "":
            res_output[head] = text_sec  

    for head, sec_text in res_output.items():
        res_paragraphs += f"Section: {head}\n"
        res_paragraphs += f"{sec_text.strip()}\n"    
    
    return res_paragraphs


def normalize_query(query: str) -> str:
    return query.lower().strip()

def retrieval_context_2wiki(search_kw, sample, embedding_model, vec_embed=False, top_k=1, cos_threshold=0.7, fuzz_threshold=0.7, vecembed_threshold=0.45):
    search_kw = search_kw.lower()
    all_context = sample['context']   
    high_score_matches = []
    fuzz_fallback_matches = []
    result = ""
    
    for item in all_context:
        key_context = item[0].lower()
        scores = caculate_scores(search_kw, key_context, embedding_model)
        if scores['cosine'] >= cos_threshold:
            high_score_matches.append({'score': scores['cosine'], 'context_item': item})
        if scores['fuzz'] >= fuzz_threshold:
            fuzz_fallback_matches.append({'score': scores['fuzz'], 'context_item': item})

    best_match_item = None
    if high_score_matches:
        best_match_item = max(high_score_matches, key=lambda x: x['score'])['context_item']
    elif fuzz_fallback_matches:
        best_match_item = max(fuzz_fallback_matches, key=lambda x: x['score'])['context_item']
    else:
        for item in all_context:
            key_context = normalize_text(item[0].lower())
            scores = caculate_scores(search_kw, key_context, embedding_model)
            
            if search_kw in key_context:
                high_score_matches.append({'score': scores['cosine'], 'context_item': item})
            elif key_context in search_kw:
                high_score_matches.append({'score': scores['cosine'], 'context_item': item})
            elif scores['fuzz'] >= 0.5:
                fuzz_fallback_matches.append({'score': scores['fuzz'], 'context_item': item})
            elif scores['cosine'] >= 0.5:
                high_score_matches.append({'score': scores['cosine'], 'context_item': item})
                
        if high_score_matches:
            best_match_item = max(high_score_matches, key=lambda x: x['score'])['context_item']
        elif fuzz_fallback_matches:
            best_match_item = max(fuzz_fallback_matches, key=lambda x: x['score'])['context_item']
                
    if best_match_item:
        key_entities = best_match_item[0]
        result += key_entities + ": "
        for sent in best_match_item[1]:
            result += sent + " "
        result += "\n\n" 
    else:
        key_entities = None
    
    # Vector Embedding (Semantic Search)
    if vec_embed:
        query_vec = embedding_model.encode(search_kw)
        for item in all_context:
            title = item[0]
            sentences = item[1]
            if title == key_entities:
                continue
            
            sent_vecs = embedding_model.encode(sentences)
            sims = cosine_similarity([query_vec], sent_vecs)[0]
            
            sentences_with_scores = []
            for i, sent in enumerate(sentences):
                sentences_with_scores.append({
                    "sent": sent,
                    "score": sims[i],
                    "pos": i
                })
                
            sorted_sentences = sorted(sentences_with_scores, key=lambda x: x["score"], reverse=True)
            top_k_sentences = [s for s in sorted_sentences if s["score"] >= vecembed_threshold][:top_k]
            
            if top_k_sentences:
                # Lưu các chỉ số của câu cần lấy
                indices_to_extract = set()
                for s_info in top_k_sentences:
                    pos = s_info["pos"]
                    indices_to_extract.add(pos)
                    if pos > 0:
                        indices_to_extract.add(pos - 1)
                    if pos < len(sentences) - 1:
                        indices_to_extract.add(pos + 1)
                
                sorted_indices = sorted(list(indices_to_extract))
                selected_sents_text = [sentences[i] for i in sorted_indices]
                
                result += f"{title}: "
                result += " ".join(selected_sents_text) + "\n\n"
                
    return result.strip()

def crawl_wiki_page_general(
    search_kw_subq,
    article_iterable,
    get_hash_id_func,
    get_link_func,
    get_title_func,
    cache_crawl_base,
    cache_filter_base,
    pre_processing_func,
    target_answer_type=None,
    embedding_model=None,
    embeding_model_cluster=None,
    cache_embed_doc={},
    topk=3,
    max_token=300,
    percentage=0.5,
    get_ner=False
):
    if embedding_model:
        type_embed = "embedding"
    else:
        type_embed = "tfidf"

    if get_ner:
        src = "CRAWL_NER_WIKI"
    else:
        src = "CRAWL_WIKI"
        
    cnt_wiki = 0
    list_doc_embed = []

    for article in article_iterable:

        hash_id = get_hash_id_func(article)
        link = get_link_func(article).lower()
        title = get_title_func(article)
        
        if 'list of' in link or 'lists of' in link or 'wikiroject' in link or 'wikipedia talk' in link:
            continue

        if not hash_id or not link:
            continue

        file_crawl = f"{cache_crawl_base}/{hash_id}"
        file_filter = f"{cache_filter_base}/{hash_id}"
        
        crawled_data = None
        raw_context = None
        filter_context = None
        # print(file_crawl)
        if os.path.exists(file_crawl):
            crawl_wiki = load_json(file_crawl)
            crawled_data = crawl_wiki.get(
                'crawl_wiki', {}
            ).get(
                'html_page', {}
            ).get(link, None)
        else:
            print(f"Crawling Wikipedia page: {link}")
            crawled_data = crawl_wikipedia_page(link)
            crawled_data_output = {
                'crawl_wiki': {
                    'html_page': {
                        link: crawled_data
                    }
                }
            }
            save_json(crawled_data_output, file_crawl)

        # Extract RAW content
        if crawled_data and 'content' in crawled_data:
            raw_context = " ".join(crawled_data['content'])

        if os.path.exists(file_filter):
            data_filter = load_json(file_filter)
            filter_context = data_filter.get(
                'crawl_wiki', {}
            ).get(
                'filter', {}
            ).get(
                link, {}
            ).get(
                type_embed, None
            )

        # Nếu chưa có filter → generate
        if filter_context is None and raw_context:

            wiki_content = pre_processing_func(crawled_data)

            if wiki_content:
                filter_context = get_paragraphs_wiki_result(
                    link,
                    wiki_content,
                    search_kw_subq,
                    target_answer_type=target_answer_type,
                    embedding_model=embedding_model,
                    max_docs_token=max_token,
                    percentage=percentage
                )

                filter_output = {
                    'crawl_wiki': {
                        'filter': {
                            link: {
                                type_embed: filter_context
                            }
                        }
                    }
                }

                save_json(filter_output, file_filter)

        if filter_context:

            cnt_wiki += 1

            if get_ner:
                docs= (
                    f"- Reference: [{src}_{cnt_wiki}] "
                    f"- Information about: {title} - {article.get('ner','')}\n"
                    f"{filter_context}\n"
                )
            else:
                docs = (
                    f"- Reference: [{src}_{cnt_wiki}] "
                    f"- Information about: {title}\n"
                    f"{filter_context}\n"
                )

            if hash_id in cache_embed_doc:
                embed_doc = cache_embed_doc[hash_id]
            else:
                embed_doc = get_embedding(filter_context, embeding_model_cluster)
            list_doc_embed.append({
                'docs': docs,
                'embedding': embed_doc,
                'hash_id': hash_id,
                'source': src
            })
            
            if cnt_wiki >= topk:
                break

    return list_doc_embed


def crawl_wiki_page_raw(
    article_iterable,
    get_hash_id_func,
    get_link_func,
    cache_crawl_base,
    topk=3
):
    all_context = ""
    cnt_wiki = 0

    for article in article_iterable:
        hash_id = get_hash_id_func(article)
        link = get_link_func(article)

        if not hash_id or not link:
            continue

        file_crawl = f"{cache_crawl_base}/{hash_id}"

        if os.path.exists(file_crawl):
            crawl_wiki = load_json(file_crawl)
            crawled_data = crawl_wiki.get('crawl_wiki', {}).get('html_page', {}).get(link, None)
            if crawled_data:
                content = crawled_data.get("content", [])
                all_context +=  " ".join(content) + "\n"
                cnt_wiki += 1
                if cnt_wiki >= topk:
                    break
        else:
            continue

    return all_context

def get_sparql_retrieval_context(list_ner_q, search_kw_subq, settings, embedding_model, 
                                 retrieval_topk, cache_hashid_sparql, file_cache_hash_id_sparql):
    """
    Xử lý lấy context từ SPARQL với logic hash_id = hash(ner + search_kw_subq)
    """
    folder_hash_sparql = settings['cache']['crawl_wikidata']
    os.makedirs(folder_hash_sparql, exist_ok=True)
    
    # 1. Lấy relation IDs dựa trên sub-query (dùng chung cho tất cả NER trong sub-query này)
    dict_relation_id = match_relation_id(
        search_kw_subq, 
        top_k=retrieval_topk['sparql_r_topk'], 
        retry_topk=retrieval_topk['sparql_r_retry_topk']
    )
    
    desc_sims = []

    for ner in list_ner_q:
        ner_norm = normalize_query(ner)
        unique_key = f"{ner_norm.strip().lower()}_{search_kw_subq.strip().lower()}"
        
        if unique_key in cache_hashid_sparql:
            hash_file_name = cache_hashid_sparql[unique_key]
        else:
            hash_file_name = make_cache_filename(unique_key)
            cache_hashid_sparql[unique_key] = hash_file_name
            save_json(cache_hashid_sparql, file_cache_hash_id_sparql)
        
        file_crawl = f"{folder_hash_sparql}/{hash_file_name}"
        
        if os.path.exists(file_crawl):                
            crawled_sparql = load_json(file_crawl)
            data = crawled_sparql.get(unique_key, {})
            context = data.get('context', "")
            description = data.get('description', "")
        else:
            # Tiến hành crawl/filter từ Wikidata API
            context, dict_result, description = process_fillter_wikidata_api(
                ner_norm, 
                settings['get_history_old_id'], 
                dict_relation_id
            )
            crawled_sparql = {unique_key: {"description": description, "context": context, "dict_result": dict_result}}
            save_json(crawled_sparql, file_crawl)
        
        # Tính toán độ tương đồng để xếp hạng (rerank)
        cosine = 0.0
        if description and isinstance(description, str) and description.strip():
            cosine = caculate_scores(search_kw_subq, description, embedding_model)["cosine"]
            
        desc_sims.append({"key_entity": ner_norm, "cosine": cosine, "context": context})

    # Format kết quả trả về
    sparql_context = ""
    if desc_sims:
        desc_sims_sorted = sorted(desc_sims, key=lambda x: x["cosine"], reverse=True)
        formatted_list = [f"{ent['key_entity']}: {ent['context']}" for ent in desc_sims_sorted if ent["context"].strip()]
        if formatted_list:
            sparql_context = "\n" + "\n".join(formatted_list)
            
    return sparql_context

from scipy.spatial.distance import cdist

def get_top_k_docs(centroid, crawl_data, topk=3, method='euclidean'): # lấy top k từ tâm cụm search    
    if not crawl_data:
        print("⚠ Empty crawl_data")
        return [], [], []

    doc_embeddings = [doc['embedding'] for doc in crawl_data]

    if len(doc_embeddings) == 0:
        print("⚠ No embeddings found")
        return [], [], []

    doc_matrix = np.array(doc_embeddings)

    if doc_matrix.ndim != 2:
        print("⚠ doc_matrix not 2D:", doc_matrix.shape)
        return [], [], []

    centroid = np.array(centroid).reshape(1, -1)
      
    if method == 'euclidean':
        distances = cdist(centroid, doc_matrix, metric='euclidean')[0]
        top_indices = np.argsort(distances)[:topk]
        
    elif method == 'similarity':
        cosine_dist = cdist(centroid, doc_matrix, metric='cosine')[0]
        similarities = 1 - cosine_dist
        top_indices = np.argsort(similarities)[::-1][:topk]
        
    else:
        raise ValueError("Method phải là 'euclidean' hoặc 'similarity'")
        
    top_texts = [crawl_data[i]['docs'] for i in top_indices]
    top_hash_ids = [crawl_data[i]['hash_id'] for i in top_indices]
    top_sources = [crawl_data[i]['source'] for i in top_indices]

    return top_texts, top_hash_ids, top_sources


def compute_cluster_info(labels, docs, embed_docs):
    """
    Gom doc + embedding theo từng label (bao gồm cả -1) và tính centroid.

    Args:
        labels (list[int])
        docs (list[str])
        embed_docs (list[list[float]])

    Returns:
        dict[label] = {
            "centroid": list[float],
            "docs": list[str],
            "embeddings": list[list[float]]
        }
    """

    assert len(labels) == len(docs) == len(embed_docs), \
        "Labels, docs và embeddings không cùng độ dài!"

    cluster_data = {}

    for label, doc, emb in zip(labels, docs, embed_docs):

        if label not in cluster_data:
            cluster_data[label] = {
                "docs": [],
                "embeddings": []
            }

        cluster_data[label]["docs"].append(doc)
        cluster_data[label]["embeddings"].append(emb)

    for label in cluster_data:
        vectors = np.array(cluster_data[label]["embeddings"])

        if vectors.ndim == 1:
            centroid = vectors
        else:
            centroid = np.mean(vectors, axis=0)

        # Convert sang list để save JSON
        cluster_data[label]["centroid"] = centroid.tolist()

    return cluster_data

def cluster_per_qid_search_api(doc_sources, crawl_data_wikipedia, embeding_model_cluster, n_clusters=3, topk=3,method='euclidean'):
    #embed doc search api
    all_embeddings = []
    for article in doc_sources:
        embed_article = get_embedding(article, embeding_model_cluster)
        # Đảm bảo chuyển np.ndarray thành list
        if isinstance(embed_article, np.ndarray):
            embed_article = embed_article.tolist()
        all_embeddings.append(embed_article)
        
    #KMEAN
    if len(all_embeddings) == 0:
        return None

    embeddings = np.array(all_embeddings)
    # giảm K nếu N nhỏ
    k = min(n_clusters, len(embeddings))

    if k < 2:
        labels = [0] * len(embeddings)
           
    else:  
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(embeddings)
        labels = labels.tolist()
        
    cluster_info = compute_cluster_info(labels, doc_sources, embeddings)
    output_dict = {}
    for cluster_label, info in cluster_info.items():
        centroid = info["centroid"]
        cluster_docs = info["docs"]

        crawl_docs, top_hash_ids, top_sources = get_top_k_docs(
            centroid=centroid,
            crawl_data=crawl_data_wikipedia,
            topk=topk,
            method=method
        )

        output_dict[cluster_label] = {
            "docs": cluster_docs,
            "crawl_doc": crawl_docs,
            'hash_id': top_hash_ids,
            'source': top_sources
        }

    return output_dict

def kmean_context_2wiki(list_article, cache_ctx_2wiki_embed_doc, embeding_model_cluster, n_cluster=3):
        """
        Phân cụm KMeans cho các doc trong list_article với k=3.
        - list_article: danh sách các doc (chuỗi)
        - cache_ctx_2wiki_embed_doc: dict chứa embedding đã tính cho từng doc (hash_id)
        - embeding_model_cluster: model embedding
        Trả về dict: {label: {docs, embeddings, centroid}}
        """
        docs = []
        embeddings = []

        for idx, item in enumerate(list_article, 1):
            # item: tuple (title, sentences)
            title = str(item[0]).strip().lower()
            sentences = item[1]
            full_doc = " ".join(sentences)
            hash_id = make_cache_filename(title)
            
            ref_doc = f"-Reference: [DOC {idx}] - Information about: {title}: {full_doc}"
            docs.append(ref_doc)
            
            # Lấy embedding từ cache nếu có, nếu không thì tính
            if hash_id in cache_ctx_2wiki_embed_doc:
                emb = cache_ctx_2wiki_embed_doc[hash_id]["embedding"]
            else:
                emb = embeding_model_cluster.encode([full_doc])[0]
                emb = emb.tolist() if isinstance(emb, np.ndarray) else emb
                
            embeddings.append(emb)

        embeddings_np = np.array(embeddings)
        k = min(n_cluster, len(embeddings_np))
        
        if k < 2:
            labels = [0] * len(embeddings_np)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_np)
            labels = labels.tolist()

        # Gom doc theo từng label, chỉ trả về docs cho mỗi cluster
        cluster_docs = {}
        for label, doc in zip(labels, docs):
            if label not in cluster_docs:
                cluster_docs[label] = []
            cluster_docs[label].append(doc)

        return cluster_docs
    

def retrieve_evidence(list_ner_q, sample, search_kw_subq, settings,target_answer_type,
        embedding_model, embeding_model_cluster, hash_id_cache, hash_id_crawl_wiki_cache, cache_ctx_2wiki_embed_doc=None):    
    retrieval_topk = settings['retrieval_topk']
    max_results_gg = retrieval_topk['max_results_search_api_gg']
    max_results_wiki = retrieval_topk['max_results_search_api_wiki']
    hash_id, hash_id_cache = register_query_cache(search_kw_subq, hash_id_cache, file_cache_index=settings['cache']['hash_id']['retrieval'])
    cache_filepath_retrieval = f"{settings['cache']['retrieval']}/{hash_id}" 
    
    
    list_article_search = []
    
    if settings['get_context_2wiki']: #Only 2wiki dataset
        # list_article_2wiki = retrieval_context_2wiki(search_kw_subq, sample, embedding_model)
        cluster_kmean_ctx_2wiki = kmean_context_2wiki(sample['context'], cache_ctx_2wiki_embed_doc, embeding_model_cluster, n_cluster=3)
        return cluster_kmean_ctx_2wiki
    
    if settings['get_gg']:
        print("Retrieval_path: ", cache_filepath_retrieval)
        results_gg_search = retrieval_google(search_kw_subq, cache_filepath_retrieval, max_results=max_results_gg)            
        snippet_gg, list_article_gg = format_search_results(results_gg_search, topk=retrieval_topk['google'], search_type='GG') 
        
        list_article_search += list_article_gg
        
     
    if settings['get_wiki']:
        results_wiki_search = retrieval_wiki(search_kw_subq, cache_filepath_retrieval, max_results=max_results_wiki)
    
        snippet_wiki, list_article_wiki = format_search_results(results_wiki_search, topk=retrieval_topk['wiki'], search_type='WIKI') 
        
        list_article_search += list_article_wiki
       
    

    if settings['get_crawl_wiki'] and settings['get_ner_wiki']:
        cache_embed_doc = load_json(settings['cache']['embed_crawl'])
        results_wiki_search = retrieval_wiki(search_kw_subq, cache_filepath_retrieval, max_results=max_results_wiki)
        
        crawl_wiki_context_embed = crawl_wiki_page_general(
            search_kw_subq,
            results_wiki_search,  # iterable là list kết quả search
            get_hash_id_func=lambda article: hash_id_crawl_wiki_cache.get(normalize_query(article['link']), None),#make_cache_filename(article['link'])
            get_link_func=lambda article: normalize_query(article['link']),
            get_title_func=lambda article: article.get('title', ''),
            cache_crawl_base=settings['cache']['crawl_wiki'],
            cache_filter_base=settings['cache']['filter_wiki'],
            pre_processing_func=wiki_pre_processing,
            target_answer_type=target_answer_type,
            embedding_model=embedding_model,
            embeding_model_cluster=embeding_model_cluster,
            cache_embed_doc=cache_embed_doc,
            topk=retrieval_topk['wiki_crawl'],
            percentage=None,
            max_token=retrieval_topk['crawl_max_token'],
        )
   
        cache_entity = load_json(settings['cache']['hash_id']['crawl_ner'])
        ner_articles = []

        for ner in list_ner_q:
            ner_norm = normalize_query(ner)
            article = cache_entity.get(ner_norm, [])[:retrieval_topk['ner_topk_article']]
            for item in article:
                item['ner'] = ner_norm
            ner_articles.extend(article)
        
        crawl_ner_wiki_embed = crawl_wiki_page_general(
            search_kw_subq,
            ner_articles,  # iterable là list các article dict
            get_hash_id_func=lambda article: article.get('hash_id', None),
            get_link_func=lambda article: article.get('url', None).lower(),
            get_title_func=lambda article: article.get('title', ''),
            cache_crawl_base=settings['cache']['ner_wiki'],
            cache_filter_base=settings['cache']['filter_ner_wiki'],
            pre_processing_func=wiki_pre_processing,
            target_answer_type=target_answer_type,
            embedding_model=embedding_model,
            embeding_model_cluster=embeding_model_cluster,
            cache_embed_doc=cache_embed_doc,
            topk=1000, # không giới hạn article ner
            max_token=retrieval_topk['crawl_max_token_ner'],
            percentage=None,
            get_ner=True
        )
    
        ctx_doc_search_fond_nbh = cluster_per_qid_search_api(list_article_search,crawl_wiki_context_embed+crawl_ner_wiki_embed, embeding_model_cluster, n_clusters=3, topk=3)

        return ctx_doc_search_fond_nbh
    
    if settings["get_sparql_context"]: 
        file_cache_hash_id_sparql = settings['cache']['hash_id']['crawl_wikidata']
        # Load hoặc khởi tạo cache mapping cho SPARQL
        if os.path.exists(file_cache_hash_id_sparql):
            cache_hashid_sparql = load_json(file_cache_hash_id_sparql)
        else:
            cache_hashid_sparql = {}

        # Gọi hàm tách riêng
        sparql_info = get_sparql_retrieval_context(
            list_ner_q=list_ner_q,
            search_kw_subq=search_kw_subq,
            settings=settings,
            embedding_model=embedding_model,
            retrieval_topk=retrieval_topk,
            cache_hashid_sparql=cache_hashid_sparql,
            file_cache_hash_id_sparql=file_cache_hash_id_sparql
        )
        
    # return ctx_doc_search_fond_nbh
    return cluster_kmean_ctx_2wiki

def remove_retrieve_evidence(current_retrieval: str, current_short: str):
    if not isinstance(current_retrieval, str) or not isinstance(current_short, str):
        return current_retrieval
    if not current_retrieval or not current_short or current_short == "":
        return current_retrieval

    # Escape regex characters in short answer
    escaped_short = re.escape(current_short.strip())

    # Remove ALL occurrences (case-insensitive)
    new_retrieval = re.sub(escaped_short, "", current_retrieval, flags=re.IGNORECASE)

    # Cleanup whitespace artifacts
    new_retrieval = re.sub(r'\s{2,}', ' ', new_retrieval)
    new_retrieval = re.sub(r'\n\s*\n+', '\n\n', new_retrieval)
    new_retrieval = new_retrieval.strip()

    return new_retrieval


def remove_specific_references(long_answer: str, retrieval_information: str) -> str:
    # Tìm các reference trong long_answer, ví dụ: [WIKI_1], [WIKI_3]
    ref_match = re.search(r'Reference:\s*([^\.\n]+)', long_answer)
    if not ref_match:
        return retrieval_information  # Không có reference nào để xóa

    # Lấy danh sách các reference, ví dụ: ['WIKI_1', 'WIKI_3']
    refs = re.findall(r'\[([A-Z_0-9]+)\]', ref_match.group(1))
    if not refs:
        return retrieval_information

    # Tạo regex để xóa các dòng reference tương ứng
    pattern = r'(^|\n)-Reference: \[({})\]:.*?(?=\n-Reference: |\n*$)'.format('|'.join(refs))
    new_retrieval = re.sub(pattern, '', retrieval_information, flags=re.DOTALL)

    # Dọn lại whitespace
    new_retrieval = re.sub(r'\n{2,}', '\n', new_retrieval).strip()
    return new_retrieval


def remove_entity_evidence(current_retrieval: str, current_short: str):
    if not isinstance(current_retrieval, str) or not isinstance(current_short, str):
        return current_retrieval
    if not current_retrieval or not current_short or current_short == "":
        return current_retrieval

    # Escape regex characters in short answer
    escaped_short = re.escape(current_short.strip())

    # Remove ALL occurrences (case-insensitive)
    new_retrieval = re.sub(escaped_short, "", current_retrieval, flags=re.IGNORECASE)

    # Cleanup whitespace artifacts
    new_retrieval = re.sub(r'\s{2,}', ' ', new_retrieval)
    new_retrieval = re.sub(r'\n\s*\n+', '\n\n', new_retrieval)
    new_retrieval = new_retrieval.strip()

    return new_retrieval
if __name__ == "__main__":
    # query = "peutz–jeghers syndrome gene variants non-sex chromosomes stk11 gene"
    # retrieval_google(query)
    
    cache_IR20 = "data/cache/IR_20"
    file_path_cache = r"data\cache\hash_id.json"
    cache_hashid = load_json(file_path_cache)
    
    input_data = load_json()
    from tqdm import tqdm
    
    for qidx, list_searchkw in tqdm(input_data.items(), desc="Processing"):
        for search_kw in list_searchkw:
            hashid, cache_hashid = register_query_cache(search_kw, cache_hashid)
            filepath = f"{cache_IR20}/{hashid}"
            
            retrieval_google(search_kw, filepath)