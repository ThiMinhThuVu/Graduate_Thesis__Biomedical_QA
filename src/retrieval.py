from src.gpt_request import search_google
from utils.crawl_wikipage import crawl_wikipedia_page
from utils.get_abs_wiki_history import get_abs_wiki_history
from utils.process_wikidata_api import process_fillter_wikidata_api, match_relation_id
from utils.util import *
from utils.hash import register_query_cache
import re
import os
import math

def retrieval_google(search_kw, filepath, max_results=10):
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
        print(f"[API SEARCH] HẾT LƯỢT SEARCHs (attempt {attempt+1}/{max_retries})")
        print("Search kw: ", search_kw)
    else:
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
                                       
    return "\n".join(context_parts)

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

def get_paragraphs_wiki_result(link, sample, subquestion, target_answer_type=None, embedding_model=None, max_docs_token=300, percentage=0.5):    
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
                
    
    
    # THUVTM: ADD SECTION BASED ON TARGET ANSWER TYPE
    used_sections = set()
    for item in selected_idx_sentences:
        idx_doc = item["idx_doc"]
        heading = list_para_sim[idx_doc]["heading"]
        used_sections.add(normalize_section_name(heading))

   
    if target_answer_type and target_answer_type in TAT_SECTION_PRIORITY:
        wanted_sections = [
            normalize_section_name(s)
            for s in TAT_SECTION_PRIORITY[target_answer_type]
        ]

        for sec_name in wanted_sections:
            # nếu section này CHƯA xuất hiện trong golden context
            if sec_name not in used_sections:
                for idx_doc, item in list_para_sim.items():
                    if normalize_section_name(item["heading"]) == sec_name:
                        selected_idx_sentences.append({
                            "idx_doc": idx_doc,
                            "score": None,
                            "text": item["text"]
                        })
                        break  # chỉ 1 paragraph / section

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
        # if text_sec != "" and len(text_sec.split(" ")) < 1000:
        if text_sec != "":
            res_output[head] = text_sec  
        
    for head, sec_text in res_output.items():
        res_paragraphs += f"Section: {head}\n"
        res_paragraphs += f"{sec_text.strip()}\n"    
    
    return res_paragraphs

def get_paragraphs_wiki_tfidf_with_info(link, sample, subquestion, max_docs_token=300):
    list_para_text = []
    para_sections = []
    key_section = sample['sections'].keys()
    key_section = [section for section in key_section if section.lower() not in move_section_list]

    for heading, item in sample['sections'].items():
        if heading.lower() in move_section_list:
            continue
        for p in item['content']:
            para_sections.append((heading, p))
            list_para_text.append(p)

    if len(list_para_text) == 0:
        print("[crawl wiki] Link này rỗng")
        return None

    similarity = find_most_relevant_doc(subquestion, list_para_text)

    count_token = 0
    selected_idx = []
    for item in similarity:
        if count_token >= max_docs_token:
            break
        idx = item["index"]
        para = list_para_text[idx]
        selected_idx.append(idx)
        count_token += len(para.split())

    # Lấy các đoạn <info> ở section cuối cùng
    last_section = key_section[-1] if key_section else None
    info_paragraphs = []
    for i, (heading, para) in enumerate(para_sections):
        if heading == last_section and para.strip().startswith("<info>"):
            info_paragraphs.append(para.strip())

    selected_paragraphs = [list_para_text[i] for i in sorted(selected_idx)]
    selected_paragraphs.extend(info_paragraphs)
    return "\n".join(selected_paragraphs)


def normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())

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
    topk=3,
    max_token=300,
    percentage=0.5,
    get_ner=False
    ):
    if embedding_model:
        type_embed = "embedding"
    else:
        type_embed = "tfidf"

    all_context = ""
    cnt_wiki = 0

    for article in article_iterable:
        hash_id = get_hash_id_func(article)
        link = get_link_func(article)
        title = get_title_func(article)
        
        if not hash_id or not link:
            continue

        file_crawl = f"{cache_crawl_base}/{hash_id}"
        file_filter = f"{cache_filter_base}/{hash_id}"

        if os.path.exists(file_filter):
            data_filter = load_json(file_filter)
            filter_context = data_filter.get('crawl_wiki', {}).get('filter', {}).get(link, {}).get(type_embed, "")
        else:
            if os.path.exists(file_crawl):
                crawl_wiki = load_json(file_crawl)
                crawled_data = crawl_wiki.get('crawl_wiki', {}).get('html_page', {}).get(link, None)
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

            if crawled_data and 'content' in crawled_data and len(crawled_data['content']) != 0:
                wiki_content = pre_processing_func(crawled_data)
            else:
                wiki_content = None

            if wiki_content:
                filter_context = get_paragraphs_wiki_result(
                    link, wiki_content, search_kw_subq,target_answer_type=target_answer_type,
                    embedding_model=embedding_model, max_docs_token=max_token, percentage=percentage
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
            else:
                filter_context = None

        if filter_context is not None:
            cnt_wiki += 1
            if get_ner:
                all_context += f"- Reference: [CRAWL_NER_WIKI_{cnt_wiki}] - Information about: {title} - {article['ner']}\n{filter_context}\n"
            else:
                all_context += f"- Reference: [CRAWL_WIKI_{cnt_wiki}] - Information about: {title}\n{filter_context}\n"
            if cnt_wiki >= topk:
                break

    return all_context

# def crawl_wiki_page_raw(
#     article_iterable,
#     get_hash_id_func,
#     get_link_func,
#     cache_crawl_base,
#     topk=3
# ):
#     all_context = ""
#     cnt_wiki = 0

#     for article in article_iterable:
#         hash_id = get_hash_id_func(article)
#         link = get_link_func(article)

#         if not hash_id or not link:
#             continue

#         file_crawl = f"{cache_crawl_base}/{hash_id}"

#         if os.path.exists(file_crawl):
#             crawl_wiki = load_json(file_crawl)
#             crawled_data = crawl_wiki.get('crawl_wiki', {}).get('html_page', {}).get(link, None)
#             if crawled_data:
#                 sections = crawled_data.get('sections', {})
#                 section_texts = []
#                 for section_name, section_info in sections.items():
#                     contents = section_info.get('content', [])
#                     if contents:
#                         section_texts.append(f"{section_name}:\n" + "\n".join(contents))
#                 full_text = "\n\n".join(section_texts)
#                 all_context += f"Information about: {link}\n{full_text}\n-------------------\n"
#                 cnt_wiki += 1
#                 if cnt_wiki >= topk:
#                     break
#         else:
#             continue

#     return all_context

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


def retrieve_evidence(list_ner_q, sample, key_entities, search_kw_subq, settings,target_answer_type,
                      embedding_model, hash_id_cache, hash_id_crawl_wiki_cache):
    
    use_crawl_wiki_raw = False #True
    
    # key_entities: List
    retrieval_information = ""
    retrieval_topk = settings['retrieval_topk']
    max_results_gg = retrieval_topk['max_results_search_api_gg']
    max_results_wiki = retrieval_topk['max_results_search_api_wiki']
    all_snippet = ""
    
    hash_id, hash_id_cache = register_query_cache(search_kw_subq, hash_id_cache, file_cache_index=settings['cache']['hash_id']['retrieval'])
    cache_filepath_retrieval = f"{settings['cache']['retrieval']}/{hash_id}" 
    print("Retrieval_path: ", cache_filepath_retrieval)
    
    if settings['get_context_2wiki']: #Only 2wiki dataset
        retrieval_information = retrieval_context_2wiki(search_kw_subq, sample, embedding_model)
            
    if settings['get_gg']:
        results_gg_search = retrieval_google(search_kw_subq, cache_filepath_retrieval, max_results=max_results_gg)            
        snippet_gg = format_search_results(results_gg_search, topk=retrieval_topk['google'], search_type='GG') 
        retrieval_information += snippet_gg + "\n\n"
        all_snippet += snippet_gg + "\n"
    
    if settings['get_wiki'] or settings['get_abs_wiki']:
        results_wiki_search = retrieval_wiki(search_kw_subq, cache_filepath_retrieval, max_results=max_results_wiki)
    
        if settings['get_wiki']:
            snippet_wiki = format_search_results(results_wiki_search, topk=retrieval_topk['wiki'], search_type='WIKI') 
            retrieval_information += snippet_wiki + "\n\n"
            all_snippet += snippet_wiki + "\n"
        # elif settings['get_abs_wiki']:
        #     abs_wiki = format_results_wiki_abs(results_wiki_search, hash_id_crawl_wiki_cache, settings, topk=retrieval_topk['wiki_abs'])
        #     retrieval_information += abs_wiki + "\n\n"

        if settings['get_crawl_wiki']:
            if use_crawl_wiki_raw:
                crawl_wiki_context = crawl_wiki_page_raw(
                    results_wiki_search,  # iterable là list kết quả search
                    get_hash_id_func=lambda article: hash_id_crawl_wiki_cache.get(normalize_query(article['link']), None),
                    get_link_func=lambda article: normalize_query(article['link']),
                    cache_crawl_base=settings['cache']['crawl_wiki'],
                    topk=retrieval_topk['wiki_crawl'],
                )
            else:
                crawl_wiki_context = crawl_wiki_page_general(
                    search_kw_subq,
                    results_wiki_search,  # iterable là list kết quả search
                    get_hash_id_func=lambda article: hash_id_crawl_wiki_cache.get(normalize_query(article['link']), None),
                    get_link_func=lambda article: normalize_query(article['link']),
                    get_title_func=lambda article: article.get('title', ''),
                    cache_crawl_base=settings['cache']['crawl_wiki'],
                    cache_filter_base=settings['cache']['filter_wiki'],
                    pre_processing_func=wiki_pre_processing,
                    target_answer_type=target_answer_type,
                    embedding_model=embedding_model,
                    topk=retrieval_topk['wiki_crawl'],
                    # percentage=0.44,
                    percentage=None,
                    max_token=retrieval_topk['crawl_max_token'],
                )
            
            retrieval_information += crawl_wiki_context + "\n"

    if settings['get_ner_wiki']:
        cache_entity = load_json(settings['cache']['hash_id']['crawl_ner'])
        ner_articles = []

        for ner in list_ner_q:
            ner_norm = normalize_query(ner)
            article = cache_entity.get(ner_norm, [])[:retrieval_topk['ner_topk_article']]
            for item in article:
                item['ner'] = ner_norm
            ner_articles.extend(article)
        
        if use_crawl_wiki_raw:
            crawl_ner_wiki = crawl_wiki_page_raw(
                ner_articles,  # iterable là list các article dict
                get_hash_id_func=lambda article: article.get('hash_id', None),
                get_link_func=lambda article: article.get('url', None),
                cache_crawl_base=settings['cache']['ner_wiki'],
                topk=1000 # không bị giới hạn vì đã lấy sẵn ở ner_articles rồi
            )
        else:
            crawl_ner_wiki = crawl_wiki_page_general(
                search_kw_subq,
                ner_articles,  # iterable là list các article dict
                get_hash_id_func=lambda article: article.get('hash_id', None),
                get_link_func=lambda article: article.get('url', None),
                get_title_func=lambda article: article.get('title', ''),
                cache_crawl_base=settings['cache']['ner_wiki'],
                cache_filter_base=settings['cache']['filter_ner_wiki'],
                pre_processing_func=wiki_pre_processing,
                target_answer_type=target_answer_type,
                embedding_model=embedding_model,
                topk=1000, # không giới hạn article ner
                max_token=retrieval_topk['crawl_max_token_ner'],
                # percentage=0.75,
                percentage=None,
                get_ner=True
            )
        
        retrieval_information += crawl_ner_wiki + "\n"
            

    if settings["get_sparql_context"]:
        # if os.path.exists(cache_filepath_filter_wiki):
        #     all_data_retrieval = load_json(cache_filepath_filter_wiki)
        # else:
        all_data_retrieval = {}
            
        if 'sparql' in all_data_retrieval:
            history = all_data_retrieval['sparql']
        else:
            history = {}
            
        if 'retrieval_information' in history:
            retrieval_information += history['retrieval_information']
        else:
            dict_relation_id = match_relation_id(search_kw_subq, top_k=retrieval_topk['sparql_r_topk'], retry_topk=retrieval_topk['sparql_r_retry_topk']) # trả về []
            desc_sims = []
            for key_entity in key_entities:
                context, dict_result, description = process_fillter_wikidata_api(key_entity, settings['get_history_old_id'], dict_relation_id)
                history[key_entity] = {"description": description, "context": context, "dict_result": dict_result}
                # Tính cosine giữa search_kw_subq và description 
                if description and isinstance(description, str) and len(description.strip()) > 0:
                    try:
                        # Use util.caculate_scores to handle both model types
                        cosine = caculate_scores(search_kw_subq, description, embedding_model)["cosine"]
                    except Exception:
                        cosine = 0.0
                else:
                    cosine = 0.0
                desc_sims.append({"key_entity": key_entity, "cosine": cosine, "context": context})

            # Chọn k_percent key_entity có cosine cao nhất
            if desc_sims:
                desc_sims_sorted = sorted(desc_sims, key=lambda x: x["cosine"], reverse=True)
                if len(key_entities) > 5:
                    n_select = max(1, math.ceil(len(desc_sims_sorted) * retrieval_topk['k_percent']))
                    top_entities = desc_sims_sorted[:n_select]
                else:                    
                    top_entities = desc_sims_sorted
                    
                for ent in top_entities:
                    if ent["context"] and ent["context"].strip():
                        retrieval_information += f"\n{ent['key_entity']}: {ent['context']}"
                history['desc_sims_sorted'] = desc_sims_sorted
                
            history['retrieval_information'] = retrieval_information
            all_data_retrieval['sparql'] = history
            # save_json(all_data_retrieval, cache_filepath_filter_wiki)
            
    return all_snippet, retrieval_information


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
