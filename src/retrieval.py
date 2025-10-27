from src.gpt_request import search_google
from utils.crawl_wikipage import crawl_wikipedia_page
from utils.abstract_wiki import crawl_abstract_wiki
from utils.get_abs_wiki_history import get_abs_wiki_history
from utils.crawl_sparql import find_sparql_context, select_top_documents
import json
import os
from utils.util import *
from thefuzz import fuzz
import re
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')


def retrieval_google(search_kw, step, filepath):
    if os.path.exists(filepath):
        output = load_json(filepath)
        step = str(step)
        if step in output:
            print(f"Cache hit for Google Search step {step}. Loading from file.")
            return output[step]['search_results']
    else:
        output = {}
            
    result = search_google(search_kw)     
    output[step] = {
        'search_kw': search_kw,
        "search_results": result                
    }
  
    save_json(output, filepath)        
    return result

def retrieval_wiki(search_kw, step, filepath):
    if os.path.exists(filepath):
        output = load_json(filepath)
        step = str(step)
        if step in output:
            print(f"Cache hit for Wiki Search step {step}. Loading from file.")
            return output[step]['search_results']
    else:
        output = {}
     
    result = search_google(search_kw + " site:wikipedia.org")           
    
    output[step] = {
        'search_kw': search_kw,
        "search_results": result                
    }
  
    save_json(output, filepath)        
    return result

def crawl_wiki_page(wiki_search, search_kw, wiki_cache_data, wiki_file_crawl, paragraph_file, step, topk = 3):  # Chỉ lấy 3 kết quả đầu tiên để tránh crawl quá nhiều
    step = str(step)
    if os.path.exists(paragraph_file):
        output_para = load_json(paragraph_file)
    else:
        output_para = {}    
        
    if step in output_para:
        all_context = output_para[step]["selected_paragraphs"]
        return all_context
    
    if os.path.exists(wiki_file_crawl):
        results_wiki = load_json(wiki_file_crawl)
    else:
        results_wiki = {}
        
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
            
    results_wiki[step] = results
    save_json(results_wiki, wiki_file_crawl)

    output_paragraph, all_context = get_paragraphs_wiki_result(results, search_kw)    
    output_para[step] = output_paragraph
    save_json(output_para, paragraph_file)

    return all_context


def format_search_results(search_results, topk=5):
    context_parts = []
    for i, result in enumerate(search_results):            
        if i == topk:
            break
        
        context_parts.append(f"- {result['title']} {result['snippet']}")
                                       
    return "\n".join(context_parts)


def format_search_results_wiki_abs(search_results, step, filepath, topk=5):
    context_parts = []
    cnt = 0
    if os.path.exists(filepath):
        output_wiki = load_json(filepath)
        if step in output_wiki:
            for link, item in output_wiki[step]['dict_para'].items():
                if cnt == topk:
                    break
                if "List_"  in link:
                    continue                
                context_parts.append(f"- {item['title']}: {item['snippet']} {item['abstract']}")
                cnt += 1
                
            return "\n".join(context_parts)
    else:
        output_wiki = {}
        
    output_wiki[step] = {}
    output_wiki[step]["dict_para"] = {}
    
    for i, result in enumerate(search_results):            
        if i == topk:
            break
        title = result['title']
        url = result['link']
        if "List_" in url:
            continue

        abs_wiki = get_abs_wiki_history(title)
        context_parts.append(f"- {title}: {result['snippet']} {abs_wiki}")
        
        output_wiki[step]["dict_para"][result['link']] = {'title': title,
                                                        'snippet': result['snippet'],
                                                        'abstract': abs_wiki}
    save_json(output_wiki, filepath)
    return "\n".join(context_parts)


def caculate_scores(search, golden):
    if search.lower() == golden.lower():
        return {'cosine': 1.0, 'fuzz': 1.0}
    
    # vectorizer = TfidfVectorizer()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))
    try:
        tfidf_matrix = vectorizer.fit_transform([search, golden])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except ValueError:
        cosine_sim = 0.0
    
    # Tính điểm fuzz (chuẩn hóa về thang 0-1)
    score_fuzz = fuzz.token_set_ratio(search, golden) / 100.0
    
    return {'cosine': cosine_sim, 'fuzz': score_fuzz}


def normalize_text(text):
    """Loại bỏ nội dung trong ngoặc đơn và các khoảng trắng thừa."""
    # Loại bỏ nội dung trong ngoặc đơn, ví dụ: " (composer)"
    text = re.sub(r'\s*\([^)]*\)\s*', '', text)
    # Loại bỏ khoảng trắng thừa ở đầu và cuối
    return text.strip()

def retrieval_context_2wiki(search_kw, sample, vec_embed=False, top_k=1, cos_threshold=0.7, fuzz_threshold=0.7, vecembed_threshold=0.45):
    search_kw = search_kw.lower()
    all_context = sample['context']   
    high_score_matches = []
    fuzz_fallback_matches = []
    result = ""
    
    for item in all_context:
        key_context = item[0].lower()
        scores = caculate_scores(search_kw, key_context)
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
            scores = caculate_scores(search_kw, key_context)
            
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
        for item in all_context:
            title = item[0]
            sentences = item[1]
            if title == key_entities:
                continue
            
            query_vec = model.encode(search_kw)
            sent_vecs = model.encode(sentences)
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

def retrieval_abstract_wiki(search_kw):
    abstract_text, page_title = crawl_abstract_wiki(search_kw)
    result = f"{page_title}: {abstract_text}"  
    return result

def retrieval_sparql(key_entity, search_kw):
    list_context = find_sparql_context(key_entity)
    ranked_docs = find_most_relevant_doc(search_kw, list_context)
    selected_context = select_top_documents(key_entity, ranked_docs)

    return selected_context
    

def perform_retrieval(sample, key_entity, search_kw, step, paths, settings, R7_info=None):
    """Thực hiện tất cả các bước truy xuất thông tin và trả về context."""
    retrieval_information = ""
    
    if 'wiki' in settings['dataset']: 
        if settings['get_context_2wiki'] == "True": 
            retrieval_information = retrieval_context_2wiki(key_entity, sample)

        if settings['get_wiki'] == "True":
            results_wiki_search = retrieval_wiki(key_entity, step, paths['wiki_search'])
            retrieval_information += format_search_results_wiki_abs(results_wiki_search, step, paths['paragraphs'], topk=1) + "\n\n"
         
        if settings['get_crawl_wiki'] == "True":
            crawl_wiki_context = crawl_wiki_page(
                results_wiki_search, key_entity, settings['wiki_cache'],
                paths['wiki_crawl'], paths['paragraphs'], step
            )
            retrieval_information += crawl_wiki_context
        
        if settings['get_abs_wiki'] == "True":
            retrieval_information = retrieval_abstract_wiki(key_entity)
        
        if settings['get_gg'] == "True":
            results_gg_search = retrieval_google(key_entity, step, paths['gg_search'])            
            retrieval_information += format_search_results(results_gg_search, topk=1) + "\n\n"
        
        if settings["get_sparql_context"] == "True":
            retrieval_information += retrieval_sparql(key_entity, search_kw)

        return retrieval_information
    
    if 'bioasq' in settings['dataset']:
        retrieval_information += settings['context']
    
    if settings['run'] == 'R7':
        ans_old = R7_info["ans_old"]        
        results_gg_search, status_gg = del_snippet(R7_info["results_gg_search"], ans_old)        
        results_wiki_search, status_wiki = del_snippet(R7_info["results_wiki_search"], ans_old)
        
        if status_gg or status_wiki:
            status = True            
            retrieval_information += format_search_results(results_gg_search) + "\n\n"
            retrieval_information += format_search_results(results_wiki_search) + "\n\n"
            
            crawl_wiki_context = R7_crawl_wiki_page(
                results_wiki_search, key_entity, settings['wiki_cache'],
                paths['wiki_crawl'], paths['paragraphs'], step
            )
            retrieval_information += crawl_wiki_context
        else:
            status = False
            
        return retrieval_information, status, results_gg_search ,results_wiki_search
            
    if settings['get_gg'] == "True":
        results_gg_search = retrieval_google(key_entity, step, paths['gg_search'])            
        retrieval_information += format_search_results(results_gg_search, topk=5) + "\n\n"
        
    if settings['get_wiki'] == "True":
        results_wiki_search = retrieval_wiki(key_entity, step, paths['wiki_search'])
        retrieval_information += format_search_results(results_wiki_search, topk=5) + "\n\n"
        
        if settings['get_crawl_wiki'] == "True":
            crawl_wiki_context = crawl_wiki_page(
                results_wiki_search, key_entity, settings['wiki_cache'],
                paths['wiki_crawl'], paths['paragraphs'], step
            )
            retrieval_information += crawl_wiki_context
        
    return retrieval_information


def norm_entity_with_keycontext(entity, sample, cos_threshold=0.7, fuzz_threshold=0.7):
    all_context = sample['context']   
    high_score_matches = []
    fuzz_fallback_matches = []
    
    for item in all_context:
        key_context = item[0].lower()
        scores = caculate_scores(entity, key_context)
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
            scores = caculate_scores(entity, key_context)
            
            if entity in key_context:
                high_score_matches.append({'score': scores['cosine'], 'context_item': item})
            elif key_context in entity:
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
        best_match_item = best_match_item[0]
    return best_match_item

def perform_retrieval_R12(sample, search_kw, step, paths, settings, type_context=None):
    """Thực hiện tất cả các bước truy xuất thông tin và trả về context."""
    retrieval_information = ""
    
    if 'wiki' in settings['dataset']: 
        if type_context == "vec_embed":
            retrieval_information = retrieval_context_2wiki(search_kw, sample, vec_embed=True)

        if type_context == "only_gg":
            results_gg_search = retrieval_google(search_kw, step, paths['gg_search'])            
            retrieval_information += format_search_results(results_gg_search, topk=5) + "\n\n"
                        
    return retrieval_information