import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "MAS_UETQUINTET"))

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import signal
import re
import requests
import matplotlib.pyplot as plt
from collections import Counter
from utils.crawl_wikipage import crawl_wikipedia_page
import pickle
import yaml

move_section_list = [
    "table of contents",
    "see also",
    "references",
    "external links",
    "further reading",
    "notes",
    "footnotes",
    "citations",
    "citation"
    "category:", 
    "categories",    
]

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)
    return configs

def load_pickle(file_path):
    if not os.path.exists(file_path):
        return {}
    
    with open(file_path, "rb") as f:
        sparql_cache = pickle.load(f)
        
    return sparql_cache

def save_pickle(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def extract_all_context(search_context_old):
    """
    Lấy nội dung của wiki crawl.
    """
    marker = '\n\n Wikipedia - Information about:'
    idx = search_context_old.find(marker)
    if idx == -1:
        return ''  # nếu không tìm thấy
    
    return search_context_old[idx:]

def check_in_ans_info(word, text):
    pattern = r'\b{}\b'.format(re.escape(word))
    return re.search(pattern, text) is not None

def check_word_after_comma(word, text):
    """Kiểm tra sau từ có phẩy không."""
    pattern = r'(\b{}\b)(,)'.format(re.escape(word))
    return re.search(pattern, text) is not None

def replace_word_after_comma(word, text):
    pattern = r'(\b{}\b),'.format(re.escape(word))
    # Xóa luôn từ cùng với dấu phẩy sau nó
    text_new = re.sub(pattern, '', text)
    # Chuẩn hoá khoảng trống
    text_new = re.sub(r'\s+', ' ', text_new).strip()
    return text_new

def del_snippet(gg_search, ans_old):
    if_search_gg = []
    check_snippet_gg = set()
    status = False
    for i, item in enumerate(gg_search, 1):
        snippet = item["snippet"]
        if check_in_ans_info(ans_old, snippet):            
            status = True
            if check_word_after_comma(ans_old, snippet):
                item["snippet"] = replace_word_after_comma(ans_old, snippet)
                if_search_gg.append(item)
        else:
            if_search_gg.append(item)

    return if_search_gg, status

def remove_sentence_containing_answer(ans, context):
    if context == "" or not ans:
        return context
    
    sentences = re.split(r'(?<=\.)\s*', context)
    filtered_sentences = []
    result_context = ""
    for sentence in sentences:
        if ans not in sentence.lower():
            filtered_sentences.append(sentence)
            result_context += sentence + ". "
    
    return result_context.strip()
    
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
def get_search_result(id, folder_path):
    filepath = f"{folder_path}/step_3a_{id}.json"
    print("filepath: ", filepath)
    if os.path.exists(filepath):
        with open(filepath , "r", encoding="utf-8") as f:
            return json.load(f)
        
    return None
     
def find_most_relevant_doc(input_text, documents):
    # Tạo list gồm input_text + các document
    corpus = [input_text] + documents

    # Khởi tạo TF-IDF vectorizer bỏ qua stop words
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Biến đổi corpus thành ma trận TF-IDF
    X = vectorizer.fit_transform(corpus)
    
    # Tính cosine similarity giữa input_text (vị trí 0) với các documents
    similarities = cosine_similarity(X[0:1], X[1:]).flatten()

    # Tạo danh sách các tuple (index, score, document) và sắp xếp theo score giảm dần
    ranked_docs = []
    for i, score in enumerate(similarities):
        ranked_docs.append({
            "index": i,
            "score": score,
            "text": documents[i]
        })
    
    # Sắp xếp theo điểm similarity giảm dần
    ranked_docs = sorted(ranked_docs, key=lambda x: x["score"], reverse=True)

    return ranked_docs

def wh_check(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    last_item = json_data[-1]
    short_answer = last_item["short_answer"].strip()
    if short_answer.lower() not in ["yes", "no", "yes.", "no.", "yes .", "no ."]:
        return True

    return False

def wiki_pre_processing(content):
    new_content = dict()
    new_content['title'] = content['title']
    new_content['url'] = content['url']
    new_content['sections'] = dict()

    for section in list(content["sections"].keys())[:-2]:
        new_content['sections'][section] = content["sections"][section]

    for section in list(content["sections"].keys())[-2:]:
        result = []
        section_content = content["sections"][section]["content"]

        infor_idxs = []
        for idx, tmp in enumerate(section_content):
            if "<info>" in tmp:
                infor_idxs.append(idx)

        for idx, tmp in enumerate(section_content):
            if "<info>" not in tmp:
                result.append(tmp)
            else:
                if len(infor_idxs) <= 15:
                    result.append(tmp)

        if len(result) > 0:
            new_content['sections'][section] = {
                "content": result
            }
    return new_content

def get_paragraphs_wiki_result(results, subquestion, max_docs_token=300):    
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

def setup_paths(settings, sample_id, q_i=""): #q_i có dạng "/q1"
    """Tạo thư mục và trả về một dictionary chứa tất cả các đường dẫn file cần thiết."""
    dataset = settings['dataset']
    run = settings['run']
    dirs = {
        "results": f"{dataset}results{q_i}",
        "decomp": f"{dataset}preprocessing/step1{q_i}",
        "retrieval": f"{dataset}preprocessing/step2{q_i}",
        "ner": f"{dataset}preprocessing/step3{q_i}",
        "all_ans": f"{dataset}results/all_ans",
        "evidence": f"{dataset}results/evidence"
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    paths = {
        "decomp": os.path.join(dirs["decomp"], f"step_1_{sample_id}.json"),
        "gg_search": os.path.join(dirs["retrieval"], f"step_2_gg_{sample_id}.json"),
        "wiki_search": os.path.join(dirs["retrieval"], f"step_2_wiki_{sample_id}.json"),
        "wiki_abs": os.path.join(dirs["retrieval"], f"step_2_wiki_abs_{sample_id}.json"),
        "wiki_crawl": os.path.join(dirs["retrieval"], f"step_2_crawl_wiki_{sample_id}.json"),
        "paragraphs": os.path.join(dirs["retrieval"], f"step_2_paragraphs_{sample_id}.json"),
        "ner": os.path.join(dirs["ner"], f"{run}_step_3_ner_{sample_id}.json"),
        "qi_answer": os.path.join(dirs["results"], f"{run}_step_3b_ans_{sample_id}.json"),
        "final_answer": os.path.join(dirs["all_ans"], f"{run}_step_3b_all_ans_{sample_id}.json"),
        "evidence": os.path.join(dirs["evidence"], f"{run}_evidence_{sample_id}.json"),
        "ans_old": os.path.join(dirs["all_ans"], f"R11_step_3b_all_ans_{sample_id}.json"),
    }
    return paths

def plot_k_context(list_k_context):
    count_dict = Counter(list_k_context)

    contexts = sorted(count_dict.keys())
    frequencies = [count_dict[c] for c in contexts]

    plt.figure(figsize=(8, 6)) 
    bars = plt.bar(contexts, frequencies)
    plt.xlabel("Số context")
    plt.ylabel("Số lượng sub-question")
    plt.title("Phân bố số lượng context")
    plt.xticks(contexts) 
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center') # va='bottom' để số nằm trên cột

    plt.show()
    
    

def R7_crawl_wiki_page(wiki_search, search_kw, wiki_cache_data, wiki_file_crawl, paragraph_file, step, topk = 3):  # Chỉ lấy 3 kết quả đầu tiên để tránh crawl quá nhiều
    step = str(step)
    results = {}    
    cnt_wiki = 0
    
    for result in wiki_search: # Chỉ lấy 3 kết quả đầu tiên để tránh crawl quá nhiều
        link = result['link']
        if 'list' in link.lower():
            continue

        cnt_wiki += 1
        if cnt_wiki > topk:
            break

        if link in wiki_cache_data:
            wiki_content = wiki_cache_data[link]
        else:
            print(f"Crawling Wikipedia page: {link}")
            wiki_content = crawl_wikipedia_page(link)

            if len(wiki_content['content']) != 0:
                wiki_content = wiki_pre_processing(wiki_content)
                wiki_cache_data[link] = wiki_content

        if wiki_content:
            results[link] = wiki_content

    output_paragraph, all_context = get_paragraphs_wiki_result(results, search_kw)    

    return all_context


class DisableKeyboardInterrupt:
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        print("Đang ghi file, chờ hoàn tất...")
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)