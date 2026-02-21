import torch
from transformers import AutoTokenizer
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

def get_embedding(text, embedding_model):
    embedding = embedding_model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True  
    )
    return embedding


question_type_to_ner_2 = {
    "condition": ["DISEASE"],
    "disease": ["DISEASE"],
    "syndrome": ["DISEASE"],
    "disorder": ["DISEASE"],
    "cancer": ["CANCER"],
    "tumor": ["CANCER"],
    "chromosome": ["DNA"],
    "gene": ["GENE_OR_GENE_PRODUCT"],
    "gene mutation": ["GENE_OR_GENE_PRODUCT"],
    "protein": ["PROTEIN"],
    "enzyme": ["PROTEIN"],
    "number of amino acids": ["PROTEIN"],
    "medication": ["CHEMICAL"],
    "drug": ["CHEMICAL"],
    "virus": ["ORGANISM"],
    "anatomical structure": ["ANATOMICAL_SYSTEM", "ORGAN", "TISSUE"],
    "muscle": ["TISSUE"],
    "nerve": ["TISSUE"]
}

question_type_to_ner_3 = {
    "condition": ["DISEASE"],
    "disease": ["DISEASE"],
    "syndrome": ["DISEASE"],
    "disorder": ["DISEASE"],
    "cancer": ["CANCER", "DISEASE"],
    "tumor": ["CANCER", "DISEASE", "TISSUE"],
    "chromosome": ["DNA", "CELLULAR_COMPONENT"],
    "gene": ["GENE_OR_GENE_PRODUCT"],
    "gene mutation": ["DNA", "GENE_OR_GENE_PRODUCT"],
    "protein": ["PROTEIN", "GENE_OR_GENE_PRODUCT"],
    "enzyme": ["PROTEIN", "GENE_OR_GENE_PRODUCT"],
    "number of amino acids": ["PROTEIN", "AMINO_ACID"],
    "medication": ["CHEMICAL", "SIMPLE_CHEMICAL"],
    "drug": ["CHEMICAL", "SIMPLE_CHEMICAL"],
    "virus": ["ORGANISM"],
    "anatomical structure": ["ANATOMICAL_SYSTEM", "ORGAN", "TISSUE"],
    "muscle": ["TISSUE", "ORGAN"],
    "nerve": ["TISSUE", "ORGAN"]
}

question_type_to_ner_4 = {
    "correctness": ["DISEASE", "CHEMICAL", "GENE_OR_GENE_PRODUCT"],
    "condition": ["DISEASE", "CHEMICAL"],
    "chromosome": ["DISEASE", "CHEMICAL"],
    "gene": ["GENE_OR_GENE_PRODUCT"],
    "disease": ["DISEASE", "CHEMICAL"],
    "medication": ["DISEASE", "CHEMICAL", "GENE_OR_GENE_PRODUCT"],
    "syndrome": ["DISEASE", "CHEMICAL"],
    "disorder": ["DISEASE"],
    "protein": ["GENE_OR_GENE_PRODUCT"],
    "person": ["DISEASE", "CHEMICAL"],
    "gene mutation": ["GENE_OR_GENE_PRODUCT"],
    "choice": ["DISEASE", "CHEMICAL"],
    "enzyme": ["GENE_OR_GENE_PRODUCT"],
    "virus": ["ORGANISM"],
    "year": ["DISEASE", "CHEMICAL"],
    "anatomical structure": ["DISEASE", "CHEMICAL"],
    "drug": ["DISEASE", "CHEMICAL", "GENE_OR_GENE_PRODUCT"],
    "symptom": ["DISEASE", "CHEMICAL", "GENE_OR_GENE_PRODUCT"],
    "cancer": ["CANCER"],
    "muscle": ["TISSUE", "ORGAN"],
    "nerve": ["TISSUE"],
    "number of amino acids": ["DISEASE", "CHEMICAL"],
    "procedure": ["DISEASE", "CHEMICAL"],
    "tumor": ["CANCER"],  
}

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

TAT_SECTION_PRIORITY = {
    "disorder": ["cause", "signs and symptoms"], #"diagnosis", 
    "gene": ["function", "genetics"],
    "disease": ["cause", "treatment"],
    "suborder": ['fossil record'],
    # "syndrome": ["introduction", "cause"],
    # "medication": ["pharmacology", "mechanism"],
    "condition": ["genetics", "risk factors"],
}

def normalize_section_name(s):
    return s.lower().strip()

#filter_ner_wikipedia
#filter_wikipedia

def is_sentence_transformer(model):
    return hasattr(model, 'encode') and 'sentence_transformers' in str(type(model))

def bert_encode(texts, model, tokenizer, device='cpu'):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        if hasattr(outputs, 'pooler_output'):
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy()

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
        if is_sentence_transformer(embedding_model):
            # print("is sentence transformer")
            embeddings = embedding_model.encode([search, golden], normalize_embeddings=True)
        else:
            # BertModel: require tokenizer and device as arguments
            # Try to get tokenizer and device from embedding_model attributes, else require as kwargs
            tokenizer = getattr(embedding_model, 'tokenizer', None)
            device = getattr(embedding_model, 'device', 'cpu')
            if tokenizer is None:
                # Try to load default tokenizer from model name
                if hasattr(embedding_model, 'name_or_path'):
                    tokenizer = AutoTokenizer.from_pretrained(embedding_model.name_or_path)
                else:
                    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            embeddings = bert_encode([search, golden], embedding_model, tokenizer, device)
        cosine_sim = float(np.dot(embeddings[0], embeddings[1]))
    except Exception:
        cosine_sim = 0.0

    fuzz_score = fuzz.token_set_ratio(search, golden) / 100.0

    return {
        "cosine": cosine_sim,
        "fuzz": fuzz_score
    }
    
def find_most_relevant_doc_embedding(input_text, documents, embedding_model):
    ranked_docs = []
    for i, doc in enumerate(documents):
        score = caculate_scores(input_text, doc, embedding_model)["cosine"]
        ranked_docs.append({
            "index": i,
            "score": float(score),
            "text": doc
        })
    ranked_docs = sorted(ranked_docs, key=lambda x: x["score"], reverse=True)
    return ranked_docs

# def find_most_relevant_doc_embedding(input_text, documents, embedding_model):
#     """
#     Tìm document liên quan nhất với input_text dựa trên SentenceTransformer embeddings.
#     """
#     # Encode input_text và documents thành embedding vectors
#     all_texts = [input_text] + documents
#     embeddings = embedding_model.encode(all_texts, convert_to_tensor=False)  # numpy array

#     # Tính cosine similarity giữa input_text (vị trí 0) với documents
#     similarities = cosine_similarity([embeddings[0]], embeddings[1:]).flatten()

#     # Tạo danh sách các document với score
#     ranked_docs = []
#     for i, score in enumerate(similarities):
#         ranked_docs.append({
#             "index": i,
#             "score": float(score),
#             "text": documents[i]
#         })
    
#     # Sắp xếp theo similarity giảm dần
#     ranked_docs = sorted(ranked_docs, key=lambda x: x["score"], reverse=True)

#     return ranked_docs

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

def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_ndarray(v) for v in obj]
    return obj

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

    section_names = list(content.get("sections", {}).keys())
    total_sec = len(section_names)
      
    if total_sec > 3:
        for section in list(content["sections"].keys())[:-2]: 
            new_content['sections'][section] = content["sections"][section]
    else:
        new_content['sections'] = content["sections"]
        
    #BỎ
    # for section in list(content["sections"].keys())[-2:]:
    #     result = []
    #     section_content = content["sections"][section]["content"]

    #     infor_idxs = []
    #     for idx, tmp in enumerate(section_content):
    #         if "<info>" in tmp:
    #             infor_idxs.append(idx)

    #     for idx, tmp in enumerate(section_content):
    #         if "<info>" not in tmp:
    #             result.append(tmp)
    #         else:
    #             if len(infor_idxs) <= 15:
    #                 result.append(tmp)

    #     if len(result) > 0:
    #         new_content['sections'][section] = {
    #             "content": result
    #         }
    return new_content

def wiki_pre_processing_tool_2(content):
    new_content = {
        "title": content["title"],
        "url": content["url"],
        "sections": {}
    }

    for section, section_data in content["sections"].items():
        passages = []

        for block in section_data.get("content", []):
            # tách theo đoạn văn: 1 hoặc nhiều dòng trống
            split_passages = [
                p.strip()
                for p in block.split("\n")
                if p.strip()
            ]
            passages.extend(split_passages)

        if passages:
            new_content["sections"][section] = {
                "content": passages
            }

    return new_content



def setup_paths(KB, settings, sample_id, q_i=""): #q_i có dạng "/q1"
    """Tạo thư mục và trả về một dictionary chứa tất cả các đường dẫn file cần thiết."""
    dataset = settings['dataset']
    run = settings['run']
    dirs = {
        "results": f"{KB}_{dataset}results{q_i}",
        "decomp": f"{dataset}preprocessing/step1{q_i}",
        "cache_decomp": f"{dataset}preprocessing/step0",
        "retrieval": f"{dataset}preprocessing/step2{q_i}",
        "ner": f"{dataset}preprocessing/step3{q_i}",
        "all_ans": f"{KB}_{dataset}results/all_ans",
        "evidence": f"{KB}_{dataset}results/evidence"
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    paths = {
        "cache_decomp": os.path.join(dirs["cache_decomp"], f"{sample_id}.json"),
        "decomp": os.path.join(dirs["decomp"], f"step_1_{sample_id}.json"),
        # "gg_search": os.path.join(dirs["retrieval"], f"step_2_gg_{sample_id}.json"),
        # "wiki_search": os.path.join(dirs["retrieval"], f"step_2_wiki_{sample_id}.json"),
        # "wiki_abs": os.path.join(dirs["retrieval"], f"step_2_wiki_abs_{sample_id}.json"),
        # "wiki_crawl": os.path.join(dirs["retrieval"], f"step_2_crawl_wiki_{sample_id}.json"),
        # "paragraphs": os.path.join(dirs["retrieval"], f"step_2_paragraphs_{sample_id}.json"),
        # "sparql": os.path.join(dirs["retrieval"], f"step_2_sparql_{sample_id}.json"),
        # "ner": os.path.join(dirs["ner"], f"{run}_step_3_ner_{sample_id}.json"),
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