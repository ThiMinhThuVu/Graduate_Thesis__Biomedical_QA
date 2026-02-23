import json
import re
import numpy as np
import os
import sys
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from postprocessing import postprocessing
import math
from src.api_key import api_key_openai
from openai import OpenAI
import portalocker # Thêm thư viện này

# ZERO SHOT
client = OpenAI(
    api_key = api_key_openai
)


def request_api_o3(messages, max_completion_tokens=8192):
    response = client.chat.completions.create(
        model="o3",
        messages=messages,
        max_completion_tokens=max_completion_tokens
    )
    return response.choices[0].message.content
    
def prompt_zero_shot_medhop_wh(question):
    prompt = f"""You are an expert in biomedical and scientific knowledge. Your task is to answer the following question based on established scientific consensus and your internal knowledge.

    ### INSTRUCTIONS:
    1. Analyze the input Question carefully to identify the key biomedical entities (e.g., proteins, genes, drugs, diseases) and their relationships.
    2. Retrieve the most relevant scientific or factual knowledge to determine the correct entity that answers the question.
    3. The short_answer MUST be the specific name of the entity or concept requested.
    4. Always provide the most accurate and well-established answer.
    5. Do NOT hedge or speculate. Provide the best possible answer.

    ### INPUT:
    Question: {question}

    ### OUTPUT FORMAT (Strict JSON):
    {{
        "short_answer": "The specific entity name (1–5 words)",
        "long_answer": "A concise explanation supporting the short_answer, mentioning key scientific facts or mechanisms."
    }}
    """
    return prompt

def prompt_zero_shot_medhop_correctness(question):
    prompt = f"""You are a biomedical expert. Your task is to answer a question base on ONLY the provided context
        ### INSTRUCTIONS:
        1. Analyze the input 'Question'. Identify key biomedical entities (e.g., drugs, proteins, diseases) and their relationships.
        2. Retrieve relevant medical facts and established scientific consensus regarding these entities.
        3. Determine the label based on the following criteria:
        - **"yes"**: The statement is factually correct and supported by established biomedical consensus.
        - **"no"**: The statement is factually incorrect or contradicted by established biomedical consensus.

        ### INPUT DATA:
        Question: {question}

        ### OUTPUT FORMAT (Strict JSON):
        {{  
            "short_answer": "yes" or "no",
            "long_answer": "[Detailed explanation based on the information]"
        }}
        """
    return prompt

CACHE_PATH = 'data/cache/embed_candidate_ans_cluster.json'

def load_embedding_cache(path):
    if not os.path.exists(path):
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return {}
    
    with open(path, "r", encoding="utf-8") as f:
        # Khóa Shared (LOCK_SH): Cho phép nhiều người đọc
        portalocker.lock(f, portalocker.LOCK_SH)
        try:
            data = json.load(f)
            # Convert list back to numpy array for calculation
            return {k: np.array(v) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError):
            return {}

def save_embedding_cache(new_data, path):
    """
    new_data: dict chứa {text: ndarray} mới phát sinh
    """
    if not new_data:
        return

    with open(path, 'r+', encoding='utf-8') as f:
        # Khóa Độc quyền (LOCK_EX): Chỉ 1 người được ghi
        portalocker.lock(f, portalocker.LOCK_EX)
        
        # 1. Đọc dữ liệu hiện tại
        f.seek(0)
        try:
            current_cache = json.load(f)
        except:
            current_cache = {}
            
        # 2. Convert dữ liệu mới (ndarray -> list) và update
        for text, emb in new_data.items():
            if isinstance(emb, np.ndarray):
                current_cache[text] = emb.tolist()
            else:
                current_cache[text] = emb
        
        # 3. Ghi đè lại file
        f.seek(0)
        f.truncate()
        json.dump(current_cache, f, ensure_ascii=False, indent=4)
          
def load_json(path):
    if not os.path.exists(path):
        # Nếu file không tồn tại, tạo file mới với nội dung {} để tránh lỗi lock file trống
        with open(path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return {}
    
    with open(path, "r", encoding="utf-8") as f:
        # Khóa Shared (LOCK_SH): Nhiều người cùng đọc được, nhưng không ai được ghi
        portalocker.lock(f, portalocker.LOCK_SH)
        data = json.load(f)
        return data

def save_json(data, file_path):
    # Dùng chế độ "a+" hoặc "r+" để portalocker hoạt động tốt nhất trên mọi OS
    with open(file_path, 'r+', encoding='utf-8') as f:
        # Khóa Độc quyền (LOCK_EX): Chỉ 1 người được quyền truy cập (cả đọc và ghi)
        portalocker.lock(f, portalocker.LOCK_EX)
        
        # Đọc dữ liệu mới nhất một lần nữa trước khi ghi 
        # (Phòng trường hợp tab kia vừa ghi xong một qidx khác)
        f.seek(0)
        try:
            current_cache = json.load(f)
        except:
            current_cache = {}
            
        # Cập nhật dữ liệu mới vào cache hiện tại
        current_cache.update(data)
        
        # Ghi đè lại file
        f.seek(0)
        f.truncate()
        json.dump(current_cache, f, ensure_ascii=False, indent=4)
        
        
def extract_references(long_answer):
    if not long_answer:
        return []
    return re.findall(r"\[([^\]]+)\]", long_answer)

def cluster_cohesion(embeddings):
    if len(embeddings) <= 1:
        return 0
    centroid = embeddings.mean(axis=0)
    dists = np.linalg.norm(embeddings - centroid, axis=1)
    return dists.mean()


def collect_candidates(raw_clusters, mapping_data):
    """
    Format dữ liệu từ cụm cluster của 1 câu hỏi thành list candidate.
    """
    cand_list = []
    
    # Duyệt qua từng cluster (cid: {llm: ..., context: ...})
    for cid, cluster in raw_clusters.items():
        if "llm" not in cluster:
            continue

        ans = cluster["llm"].get("short_answer", "").strip().lower()
        long_answer = cluster["llm"].get("long_answer", "")
        context = cluster.get("context", "").lower()
        
        # Loại bỏ các câu trả lời rác hoặc thiếu thông tin
        if not ans or "insufficient information" in ans:
            continue

        # Chuẩn hóa answer
        if ans in mapping_data:
            norm_ans = mapping_data[ans]
        else:
            norm_ans = postprocessing(ans)
            norm_ans = mapping_data.get(norm_ans, norm_ans)
        
        # Trích xuất nguồn và phân loại
        raw_sources = extract_references(long_answer)
        sources = list(set(raw_sources))
        
        has_wiki = any("WIKI" in s.upper() for s in sources)
        has_gg = any("GG" in s.upper() for s in sources)

        cand_list.append({
            "cid": cid,
            "text": norm_ans,
            "long_answer": long_answer,
            "context": context,
            "sources": sources,
            "has_wiki": has_wiki,
            "has_gg": has_gg
        })
                
    return cand_list


def log_plus(x):
    """Hàm log+(x) = log(1 + x) dùng để làm mềm ảnh hưởng outlier"""
    return math.log(1 + x)

def calculate_new_score(individual_ref_counts, nc): 
    """
    CÔNG THỨC 2B, chọn MIN = 6.5
    Tính toán điểm dựa trên công thức mới:
    S(C) = log(1 + max r_i) + sum(log(1 + r_i)) + log(1 + Nc)
    """
    # 1. Thành phần log(1 + max r_i) - Ưu tiên ứng viên mạnh nhất
    max_ri = max(individual_ref_counts) if individual_ref_counts else 0
    term_max = log_plus(max_ri)
    
    # 2. Thành phần sum(log(1 + r_i)) - Ưu tiên cụm có tổng lượng tham chiếu lớn
    term_sum_all = sum(log_plus(ri) for ri in individual_ref_counts)
    
    # 3. Thành phần log(1 + Nc) - Ưu tiên cụm có số lượng ứng viên đông
    term_nc = log_plus(nc)
    
    # Tổng hợp theo công thức mới trong ảnh
    final_score = term_max + term_sum_all + term_nc
    
    return final_score

def calculate_gini_adjusted_score(individual_ref_counts, W_MASS = 0.7, W_CONC = 0.3):
    """
    CÔNG THỨC 6: Row 36 - 37
    Gini / Concentration Adjusted Score (lambda = 1)

    S(C) = W_MASS * R_sum + W_CONC * concentration
    """

    if not individual_ref_counts:
        return 0.0

    r_sum = sum(individual_ref_counts)
    if r_sum == 0:
        return 0.0

    # concentration = sum( (ri / r_sum)^2 )
    concentration = sum((ri / r_sum) ** 2 for ri in individual_ref_counts)

    return (W_MASS * r_sum) + (W_CONC * concentration)

def cluster_and_selected_candidate_YN(cand_list, MIN_SCORE_THRESHOLD=2):
    if not cand_list:
        return "not selected"

    clusters = defaultdict(list)
    for i, c in enumerate(cand_list):
        label = c["text"].strip().lower()
        clusters[label].append(i)

    raw_clusters = []
    for label, indices in clusters.items():
        individual_ref_counts = [len(cand_list[i]["sources"]) for i in indices]
        nc = len(indices)

        c_score = calculate_gini_adjusted_score(individual_ref_counts) # CT6: MIN_SCORE_THRESHOLD=2
        # c_score = calculate_new_score(individual_ref_counts, nc) #CT 2B: MIN_SCORE_THRESHOLD=6.5
        
        raw_clusters.append({
            "label": label,
            "indices": indices,
            "evidence_count": nc,
            "cluster_score": c_score
        })

    best_cluster_info = None
    if raw_clusters:
        candidate_best = max(raw_clusters, key=lambda x: x["cluster_score"])
        if candidate_best["cluster_score"] >= MIN_SCORE_THRESHOLD:
            best_cluster_info = candidate_best

    if best_cluster_info is None:
        return "not selected"

    chosen_indices = best_cluster_info["indices"]
    best_idx = max(chosen_indices, key=lambda idx: len(cand_list[idx]["sources"]))

    # 5. Build output
    result = cand_list[best_idx].copy()
    result.update({
        "short_answer": best_cluster_info["label"],
        "selection_status": "selected",
        "cluster_score": round(best_cluster_info["cluster_score"], 4),
        "total_candidates_in_cluster": len(chosen_indices)
    })

    return result


def cluster_and_selected_candidate_WH(cand_list, model, MIN_SCORE_THRESHOLD=2, DISTANCE_THRESHOLD = 0.12):
    """
    Đánh giá list candidate của 1 câu hỏi.
    Trả về candidate tốt nhất kèm metadata hoặc "not selected".
    """
    if not cand_list:
        return "not selected"

    # 1. Encoding và Clustering
    global_cache = load_embedding_cache(CACHE_PATH)
    
    texts = [c["text"] for c in cand_list]
    unique_texts = list(set(texts))
    
    # Tìm những text chưa có trong cache
    missing_texts = [t for t in unique_texts if t not in global_cache]
    
    if missing_texts:
        new_embeddings = model.encode(missing_texts)
        new_cache_chunk = dict(zip(missing_texts, new_embeddings))        
        save_embedding_cache(new_cache_chunk, CACHE_PATH)        
        global_cache.update(new_cache_chunk)
        
    # Tạo ma trận embedding cho các candidate hiện tại từ cache đã được cập nhật
    embeddings = np.array([global_cache[t] for t in texts])
    
    raw_clusters = []

    if len(cand_list) >= 2:
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=DISTANCE_THRESHOLD,
            metric="cosine", linkage="average"
        ).fit(embeddings).labels_

        for lbl in set(labels):
            indices = [i for i, l in enumerate(labels) if l == lbl]
            individual_ref_counts = [len(cand_list[i]["sources"]) for i in indices]
            nc = len(indices)
            
            c_score = calculate_gini_adjusted_score(individual_ref_counts) # CT6: MIN_SCORE_THRESHOLD=2
            # c_score = calculate_new_score(individual_ref_counts, nc) #CT 2B: MIN_SCORE_THRESHOLD=6.5
        
            raw_clusters.append({
                "indices": indices,
                "evidence_count": nc,
                "cluster_score": c_score,
                "unique_source": sum(individual_ref_counts)
            })
    else:
        # Trường hợp chỉ có 1 candidate
        ref_counts = [len(cand_list[0]["sources"])]
        raw_clusters.append({
            "indices": [0],
            "cluster_score": calculate_gini_adjusted_score(ref_counts, 1)
        })
    
    # 2. Lọc cluster theo ngưỡng MIN_SCORE_THRESHOLD
    best_cluster_info = None
    if raw_clusters:
        candidate_best = max(raw_clusters, key=lambda x: x["cluster_score"])
        if candidate_best["cluster_score"] >= MIN_SCORE_THRESHOLD:
            best_cluster_info = candidate_best

    # 3. Trả về kết quả cuối cùng
    if best_cluster_info is None:
        return "not selected"
    
    # Chọn đại diện tốt nhất trong cụm thắng cuộc (người có nhiều ref nhất)
    chosen_indices = best_cluster_info["indices"]
    # best_idx = max(chosen_indices, key=lambda idx: len(cand_list[idx]["sources"]))
    
    # # Build kết quả trả về bao gồm toàn bộ info ban đầu và thông số debug
    # result = cand_list[best_idx].copy()
    
    merged_in_cluster = {}

    for idx in chosen_indices:
        cand = cand_list[idx]
        norm_text = cand["text"]
        
        if norm_text not in merged_in_cluster:
            merged_in_cluster[norm_text] = {
                "text": cand["text"],
                "sources": set(cand.get("sources", [])),
                "context": cand.get("context", ""),
                "long_answer": cand.get('long_answer', "")
            }
        else:
            merged_in_cluster[norm_text]["sources"].update(cand.get("sources", []))
            merged_in_cluster[norm_text]["long_answer"] += cand['long_answer']

    result = max(merged_in_cluster.values(), key=lambda x: len(x["sources"]))
    result["sources"] = list(result["sources"])
    result.update({
        "short_answer": result["text"], 
        "long_answer": result["long_answer"],
        "selection_status": "selected",
        "cluster_score": round(best_cluster_info["cluster_score"], 4),
        "total_candidates_in_cluster": len(chosen_indices)
    })
    return result

def process_get_final_answer(qidx, question, qi, pathi, stepi, target_answer_type, cluster_raw, mapping_data, embedding_model, o3_cache_file):
    candidates = collect_candidates(cluster_raw, mapping_data) 
    if target_answer_type != 'correctness':
        final_output = cluster_and_selected_candidate_WH(candidates, embedding_model) 
    else:
        final_output = cluster_and_selected_candidate_YN(candidates)
        
    if final_output == 'not selected':
        cache_o3_ans = load_json(o3_cache_file) 
        
        cached_res = cache_o3_ans.get(qidx, {}).get(qi, {}).get(pathi, {}).get(stepi)
        
        if cached_res:
            print("[Load Cache O3 Answer Zeroshot]")
            return {
                "short_answer": cached_res.get("short_answer"),
                "long_answer": cached_res.get("long_answer"),
                "selection_status": "zero-shot_fallback_cached",
                "cluster_score": None
            }
                        
        if target_answer_type == 'correctness':
            prompt = prompt_zero_shot_medhop_correctness(question)            
        else:
            prompt = prompt_zero_shot_medhop_wh(question)
            
        messages = [
            {"role": "system", "content": "You are a biomedical expert. Your task is to answer a question base on ONLY the provided context"},
            {"role": "user", "content": prompt}
        ]  
        result_raw = request_api_o3(messages)
        
        try:
            # Clean markdown nếu có
            clean_json = re.sub(r"```json|```", "", result_raw).strip()
            result_dict = json.loads(clean_json)
            if qidx not in cache_o3_ans: cache_o3_ans[qidx] = {}
            if qi not in cache_o3_ans[qidx]: cache_o3_ans[qidx][qi] = {}
            if pathi not in cache_o3_ans[qidx][qi]: cache_o3_ans[qidx][qi][pathi] = {}
            
            cache_o3_ans[qidx][qi][pathi][stepi] = result_dict
            save_json(cache_o3_ans ,o3_cache_file)
            
            # Đồng bộ cấu trúc trả về cho module ngoài dễ xử lý
            return {
                "short_answer": result_dict.get("short_answer"),
                "long_answer": result_dict.get("long_answer"),
                "selection_status": "zero-shot_fallback",
                "cluster_score": None
            }
        except Exception as e:
            return {"error": f"JSON parsing failed: {str(e)}", "raw_content": result_raw}
    
    return final_output

  
def load_pubmed_model():
    print("Loading embedding model (PubMedBERT)...")
    
    return SentenceTransformer("pritamdeka/S-BioBert-snli-multinli-stsb",device="cpu")
  
    
if __name__ == '__main__':
    # Khởi tạo model một lần duy nhất
    global_model = load_pubmed_model()
    mapping = load_json('data/mapping_answer_gol.json')
    
    # Giả sử có dữ liệu mẫu
    # res = process_get_final_answer(q, cluster, mapping, global_model)
      