from src.gpt_request import search_google
from utils.crawl_wikipage import crawl_wikipedia_page
from utils.get_abs_wiki_history import get_abs_wiki_history
from utils.process_wikidata_api import process_fillter_wikidata_api, match_relation_id
from utils.util import *
from utils.hash import register_query_cache
import re
import os
import math

def retrieval_google(search_kw, filepath):
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
        result = search_google(search_kw)
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

def retrieval_wiki(search_kw, filepath):
    if os.path.exists(filepath):
        output = load_json(filepath)
        if 'wiki' in output:
            print(f"Cache hit Wikipedia Search. Loading from file.")
            return output['wiki']['search_results']
    else:
        output = {}
     
    result = search_google(search_kw + " site:wikipedia.org")           
    
    output['wiki'] = {
        'search_kw': search_kw,
        "search_results": result                
    }
  
    save_json(output, filepath)        
    return result


def format_search_results(search_results, topk=5):
    context_parts = []
    for i, result in enumerate(search_results):            
        if i == topk:
            break
        
        context_parts.append(f"- {result['title']} {result['snippet']}")
                                       
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


def get_paragraphs_wiki_result(results, subquestion, embedding_model=None, max_docs_token=300):    
    # Lấy paragraph
    dict_para = {}
    all_context = ""
    
    for link, sample in results.items():
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
                    "idx": idx,
                    "heading": heading,
                    "text": p
                }
                index_doc += 1
                list_para_text.append(p)
        
        if len(list_para_text) == 0:
            continue
                    
        if embedding_model:
            similarity = find_most_relevant_doc_embedding(subquestion, list_para_text, embedding_model)
        else:
            similarity = find_most_relevant_doc(subquestion, list_para_text)
            
        count_token = 0
        selected_idx_sentences = []
        
        for item in similarity:
            if count_token >= max_docs_token:
                break
            
            score = item["score"]
            idx = item["index"]
            para = list_para_sim[idx]["text"]
            selected_idx_sentences.append({
                "idx": idx,
                "score": score,
                "text": para,
            })
            count_token += len(para.split())
            
            # THêm hàng xóm
            index_section = list_para_sim[idx]["idx"]
            selected_idx_set = {s["idx"] for s in selected_idx_sentences}
            
            if index_section > 0:
                idx_nbr = idx-1
                if idx_nbr not in selected_idx_set:
                    nbr_para = list_para_sim[idx_nbr]["text"]
                    selected_idx_sentences.append({
                        "idx": idx_nbr,#idx của doc
                        "score": score,
                        "text": nbr_para,
                    })
                    count_token += len(nbr_para.split())
            
            if index_section < len(sample['sections'][list_para_sim[idx]["heading"]]['content']) - 1:
                idx_nbr = idx + 1
                if idx_nbr not in selected_idx_set:
                    nbr_para = list_para_sim[idx_nbr]["text"]
                    selected_idx_sentences.append({
                        "idx": idx_nbr,#idx của doc
                        "score": score,
                        "text": nbr_para,
                    })
                    count_token += len(nbr_para.split())
                    
        list_idx = [s["idx"] for s in selected_idx_sentences]

        # thêm li
        last_section = key_section[-1]
        cnt_li = 0
        list_idx_last_sect = []
        
        for idx, item in list_para_sim.items():
            if idx not in list_idx and item["text"].strip().startswith("<info>"):
                heading = item["heading"]
                if heading.lower() in move_section_list:
                    continue
                
                if heading == last_section:                    
                    cnt_li += 1
                    list_idx_last_sect.append(idx)
                       
                selected_idx_sentences.append({
                    "idx": idx,
                    "score": None,  
                    "text": item["text"],
                })
        
        # Nếu section cuối có > 30 li thì xóa các li của section cuối
        if cnt_li > 30:
            selected_idx_sentences = [item for item in selected_idx_sentences if item["idx"] not in list_idx_last_sect]
                        
        selected_idx_sentences = sorted(selected_idx_sentences, key=lambda x: x["idx"])      
        
        res_paragraphs = ""
        res_output = {}
        
        # xếp theo thứ tự của doc, section
        for head in key_section:
            text_sec = ""
            for item in selected_idx_sentences:
                idx = item["idx"]
                text = item["text"]
                heading = list_para_sim[idx]["heading"]
                
                if head == heading:
                    if item["text"].strip().startswith("<info>"):
                        text = text.replace("<info>", "", 1).strip()
                    text_sec += f"{text.strip()} "
            if text_sec != "" and len(text_sec.split(" ")) < 1000:
                res_output[head] = text_sec      
                        
        for head, sec_text in res_output.items():
            res_paragraphs += f"Section: {head}\n"
            res_paragraphs += f"{sec_text.strip()}\n"
          
        dict_para[link] = {
            "paragraphs": res_paragraphs.strip(),
        }
        
        all_context += "Information about: {}".format(sample["title"]) + "\n" +  res_paragraphs + "-------------------"
        
    output_paragraph = {
        "dict_para": dict_para,
        "selected_paragraphs": all_context,
    }
    
    return output_paragraph, all_context


def crawl_wiki_page(wiki_search, search_kw, wiki_cache_data, cache_filepath_filter_wiki, embedding_model, topk=3, max_token=300):  # Chỉ lấy 3 kết quả đầu tiên để tránh crawl quá nhiều
    if os.path.exists(cache_filepath_filter_wiki):
        output = load_json(cache_filepath_filter_wiki)
    else:
        output = {}
        
    if 'crawl_wiki' in output:  
        output_para = output['crawl_wiki'] #selected_paragraphs; html_page
    else:
        output_para = {}    
        
    if "filter" in output_para:
        print("[Load cache] - crawl wiki page")
        all_context = output_para["filter"].get("selected_paragraphs", '')
        return all_context
       
    results = {}    
    cnt_wiki = 0
    
    for result in wiki_search: # Chỉ lấy 3 kết quả đầu tiên để tránh crawl quá nhiều
        link = result['link']
        if 'list' in link.lower():
            continue

        if cnt_wiki == topk:
            break
        
        wiki_content = None
        if link in wiki_cache_data:
            wiki_content = wiki_cache_data[link]
        else:
            print(f"Crawling Wikipedia page: {link}")
            crawled_data = crawl_wikipedia_page(link)

            if crawled_data and 'content' in crawled_data and len(crawled_data['content']) != 0:
                wiki_content = wiki_pre_processing(crawled_data)
                # wiki_cache_data[link] = wiki_content

        if wiki_content:
            results[link] = wiki_content
            cnt_wiki += 1
            
    output_para['html_page'] = results

    output_paragraph, all_context = get_paragraphs_wiki_result(results, search_kw, embedding_model, max_token)    
    output_para['filter'] = output_paragraph
    
    output['crawl_wiki'] = output_para
    save_json(output, cache_filepath_filter_wiki)

    return all_context


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


def retrieve_evidence(sample, key_entities, search_kw_subq, settings, embedding_model, hash_id_cache):
    # key_entities: List
    retrieval_information = ""
    retrieval_topk = settings['retrieval_topk']
    all_snippet = ""
    
    hash_id, hash_id_cache = register_query_cache(search_kw_subq, hash_id_cache)
    cache_filepath_retrieval = f"{settings['cache']['retrieval']}/{hash_id}" 
    cache_filepath_filter_wiki = f"{settings['cache']['filter_wiki']}/{hash_id}" # wikipedia + wikidata
    print("Retrieval_path: ", cache_filepath_retrieval)
    print("Crawl wiki path: ", cache_filepath_filter_wiki)
    
    if settings['get_context_2wiki']: #Only 2wiki dataset
        retrieval_information = retrieval_context_2wiki(search_kw_subq, sample, embedding_model)
            
    if settings['get_gg']:
        results_gg_search = retrieval_google(search_kw_subq, cache_filepath_retrieval)            
        snippet_gg = format_search_results(results_gg_search, topk=retrieval_topk['google']) 
        retrieval_information += snippet_gg + "\n\n"
        all_snippet += snippet_gg + "\n"
    
    if settings['get_wiki'] or settings['get_abs_wiki']:
        results_wiki_search = retrieval_wiki(search_kw_subq, cache_filepath_retrieval)
    
        if settings['get_wiki']:
            snippet_wiki = format_search_results(results_wiki_search, topk=retrieval_topk['wiki']) 
            retrieval_information += snippet_wiki + "\n\n"
            all_snippet += snippet_wiki + "\n"
        elif settings['get_abs_wiki']:
            abs_wiki = format_results_wiki_abs(results_wiki_search, cache_filepath_filter_wiki, topk=retrieval_topk['wiki_abs'])
            retrieval_information += abs_wiki + "\n\n"

        if settings['get_crawl_wiki']:
            crawl_wiki_context = crawl_wiki_page(
                results_wiki_search, search_kw_subq, settings['wiki_cache'], cache_filepath_filter_wiki, 
                embedding_model, topk=retrieval_topk['wiki_crawl'], max_token=retrieval_topk['crawl_max_token'])
            retrieval_information += "- Wikipedia: " + crawl_wiki_context

    if settings["get_sparql_context"]:
        if os.path.exists(cache_filepath_filter_wiki):
            all_data_retrieval = load_json(cache_filepath_filter_wiki)
        else:
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
            save_json(all_data_retrieval, cache_filepath_filter_wiki)
            
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
