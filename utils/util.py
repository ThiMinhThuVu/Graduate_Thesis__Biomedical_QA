import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "MAS_UETQUINTET"))

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import signal
import re
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import yaml
from thefuzz import fuzz
import numpy as np

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

def caculate_scores(search, golden, embedding_model, is_tfidf = False):
    if search.strip().lower() == golden.strip().lower():
        return {"cosine": 1.0, "fuzz": 1.0}

    if is_tfidf:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5)
        )
        try:
            tfidf_matrix = vectorizer.fit_transform([search, golden])
            cosine_sim = cosine_similarity(
                tfidf_matrix[0:1],
                tfidf_matrix[1:2]
            )[0][0]
        except ValueError:
            cosine_sim = 0.0

        fuzz_score = fuzz.token_set_ratio(search, golden) / 100.0

        return {
            "cosine": float(cosine_sim),
            "fuzz": float(fuzz_score)
        }

    try:
        embeddings = embedding_model.encode(
            [search, golden],
            normalize_embeddings=True
        )
        cosine_sim = float(np.dot(embeddings[0], embeddings[1]))
    except Exception:
        cosine_sim = 0.0

    fuzz_score = fuzz.token_set_ratio(search, golden) / 100.0

    return {
        "cosine": cosine_sim,
        "fuzz": fuzz_score
    }
    
def find_most_relevant_doc_embedding(input_text, documents, embedding_model):
    """
    Tìm document liên quan nhất với input_text dựa trên SentenceTransformer embeddings.
    """
    # Encode input_text và documents thành embedding vectors
    all_texts = [input_text] + documents
    embeddings = embedding_model.encode(all_texts, convert_to_tensor=False)  # numpy array

    # Tính cosine similarity giữa input_text (vị trí 0) với documents
    similarities = cosine_similarity([embeddings[0]], embeddings[1:]).flatten()

    # Tạo danh sách các document với score
    ranked_docs = []
    for i, score in enumerate(similarities):
        ranked_docs.append({
            "index": i,
            "score": float(score),
            "text": documents[i]
        })
    
    # Sắp xếp theo similarity giảm dần
    ranked_docs = sorted(ranked_docs, key=lambda x: x["score"], reverse=True)

    return ranked_docs

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

def normalize_text(text):
    """Loại bỏ nội dung trong ngoặc đơn và các khoảng trắng thừa."""
    # Loại bỏ nội dung trong ngoặc đơn, ví dụ: " (composer)"
    text = re.sub(r'\s*\([^)]*\)\s*', '', text)
    # Loại bỏ khoảng trắng thừa ở đầu và cuối
    return text.strip()

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
        "sparql": os.path.join(dirs["retrieval"], f"step_2_sparql_{sample_id}.json"),
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