import json
import re
import numpy as np
import os
import sys
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from postprocessing import postprocessing
import math
from src.api_key import api_key_openai
from openai import OpenAI
import portalocker # Thêm thư viện này


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
    prompt = f"""You are an expert in biomedical and scientific knowledge. Your task is to answer the following question based on established scientific consensus and your internal knowledge.
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
                  
def load_json(path, timeout=60):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return {}

    # Lock để đọc (LOCK_SH: nhiều người cùng đọc được)
    try:
        with portalocker.Lock(path, mode='r', timeout=timeout, encoding='utf-8') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Lỗi load file {path}: {e}")
        return {}

def save_json(data, file_path, timeout=60):
    # Sử dụng LOCK_EX (Độc quyền) để đảm bảo không ai đọc/ghi khi ta đang cập nhật
    try:
        with portalocker.Lock(file_path, mode='r+', timeout=timeout, encoding='utf-8') as f:
            # 1. Đọc dữ liệu mới nhất trước khi cập nhật (Tránh ghi đè mất dữ liệu của tiến trình khác)
            try:
                current_data = json.load(f)
            except:
                current_data = {}
            
            # 2. Cập nhật
            current_data.update(data)
            
            # 3. Ghi đè
            f.seek(0)
            f.truncate()
            json.dump(current_data, f, ensure_ascii=False, indent=4)
            # File sẽ tự động unlock khi thoát khỏi block 'with'
    except Exception as e:
        print(f"Không thể ghi file {file_path} sau {timeout}s: {e}")
        
# def load_embedding_cache(path):
#     if not os.path.exists(path):
#         # Đảm bảo thư mục tồn tại
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump({}, f)
#         return {}
    
#     with open(path, "r", encoding="utf-8") as f:
#         # Khóa Shared (LOCK_SH): Cho phép nhiều người đọc
#         portalocker.lock(f, portalocker.LOCK_SH)
#         try:
#             data = json.load(f)
#             # Convert list back to numpy array for calculation
#             return {k: np.array(v) for k, v in data.items()}
#         except (json.JSONDecodeError, ValueError):
#             return {}

# def save_embedding_cache(new_data, path):
#     """
#     new_data: dict chứa {text: ndarray} mới phát sinh
#     """
#     if not new_data:
#         return

#     with open(path, 'r+', encoding='utf-8') as f:
#         # Khóa Độc quyền (LOCK_EX): Chỉ 1 người được ghi
#         portalocker.lock(f, portalocker.LOCK_EX)
        
#         # 1. Đọc dữ liệu hiện tại
#         f.seek(0)
#         try:
#             current_cache = json.load(f)
#         except:
#             current_cache = {}
            
#         # 2. Convert dữ liệu mới (ndarray -> list) và update
#         for text, emb in new_data.items():
#             if isinstance(emb, np.ndarray):
#                 current_cache[text] = emb.tolist()
#             else:
#                 current_cache[text] = emb
        
#         # 3. Ghi đè lại file
#         f.seek(0)
#         f.truncate()
#         json.dump(current_cache, f, ensure_ascii=False, indent=4)
          
# def load_json(path):
#     if not os.path.exists(path):
#         # Nếu file không tồn tại, tạo file mới với nội dung {} để tránh lỗi lock file trống
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump({}, f)
#         return {}
    
#     with open(path, "r", encoding="utf-8") as f:
#         # Khóa Shared (LOCK_SH): Nhiều người cùng đọc được, nhưng không ai được ghi
#         portalocker.lock(f, portalocker.LOCK_SH)
#         data = json.load(f)
#         return data

# def save_json(data, file_path):
#     # Dùng chế độ "a+" hoặc "r+" để portalocker hoạt động tốt nhất trên mọi OS
#     with open(file_path, 'r+', encoding='utf-8') as f:
#         # Khóa Độc quyền (LOCK_EX): Chỉ 1 người được quyền truy cập (cả đọc và ghi)
#         portalocker.lock(f, portalocker.LOCK_EX)
        
#         # Đọc dữ liệu mới nhất một lần nữa trước khi ghi 
#         # (Phòng trường hợp tab kia vừa ghi xong một qidx khác)
#         f.seek(0)
#         try:
#             current_cache = json.load(f)
#         except:
#             current_cache = {}
            
#         # Cập nhật dữ liệu mới vào cache hiện tại
#         current_cache.update(data)
        
#         # Ghi đè lại file
#         f.seek(0)
#         f.truncate()
#         json.dump(current_cache, f, ensure_ascii=False, indent=4)
        
        
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

from difflib import SequenceMatcher

def get_similarity(a, b):
    # Tính toán độ tương đồng giữa 2 chuỗi (0.0 đến 1.0)
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

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
       
        cand_list.append({
            "cid": cid,
            "text": norm_ans,
            "raw_ans": ans,
            "long_answer": long_answer,
            "context": context,
            "sources": sources,
        })
                
    return cand_list


def log_plus(x):
    """Hàm log+(x) = log(1 + x) dùng để làm mềm ảnh hưởng outlier"""
    return math.log(1 + x)

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

def sigmoid(x):
    """Hàm Sigmoid theo ảnh: 1 / (1 + e^-x)"""
    return 1 / (1 + math.exp(-x))

def calculate_concentration(individual_ref_counts):
    """
    Tính 1 - G theo ảnh: 1 - (1 - sum(pi^2)) = sum(pi^2)
    Đây chính là mức độ tập trung (Concentration).
    """
    r_sum = sum(individual_ref_counts)
    if r_sum == 0:
        return 0.0
    # pi = ri / r_sum
    # 1 - G = sum(pi^2)
    return sum((ri / r_sum) ** 2 for ri in individual_ref_counts)

# công thức mới 2 anh AN:
def score_v2_log(individual_ref_counts, w_s=0.7, w_c=0.3):
    """CT 2: Score = w_s * log(1 + R_total) + w_c * (1 - G)"""
    r_total = sum(individual_ref_counts)
    one_minus_g = calculate_concentration(individual_ref_counts)
    return (w_s * math.log(1 + r_total)) + (w_c * one_minus_g)

def score_v3_sigmoid(individual_ref_counts, w_s=0.7, w_c=0.3):
    """CT 3: Score = w_s * sigmoid(R_total) + w_c * (1 - G)"""
    r_total = sum(individual_ref_counts)
    one_minus_g = calculate_concentration(individual_ref_counts)
    # Norm R_total bằng sigmoid
    return (w_s * sigmoid(r_total)) + (w_c * one_minus_g)

def cluster_and_selected_candidate_YN(cand_list, MIN_SCORE_THRESHOLD=2):
    if not cand_list:
        return {"result": "not selected", "statistics": {}, "selected_cluster_id": None}

    clusters = defaultdict(list)
    for i, c in enumerate(cand_list):
        label = c["text"].strip().lower()
        clusters[label].append(i)

    cluster_statistics = {}
    raw_clusters = []

    for label, indices in clusters.items():
        individual_ref_counts = [len(cand_list[i]["sources"]) for i in indices]
        
        # Tính 3 loại score
        ct1 = calculate_gini_adjusted_score(individual_ref_counts)
        ct2 = score_v2_log(individual_ref_counts)
        ct3 = score_v3_sigmoid(individual_ref_counts)

        # Lưu thống kê chi tiết
        cluster_statistics[label] = {
            "num_candidates": len(indices),
            "candidate_evidences": {idx: len(cand_list[idx]["sources"]) for idx in indices},
            "scores": {
                "ct_1": round(ct1, 4),
                "ct_2": round(ct2, 4),
                "ct_3": round(ct3, 4)
            }
        }

        raw_clusters.append({
            "label": label,
            "indices": indices,
            "cluster_score": ct1 # Dùng CT1 để chọn Best
        })

    best_cluster_info = None
    selected_cluster_id = None
    if raw_clusters:
        candidate_best = max(raw_clusters, key=lambda x: x["cluster_score"])
        if candidate_best["cluster_score"] >= MIN_SCORE_THRESHOLD:
            best_cluster_info = candidate_best
            selected_cluster_id = candidate_best["label"]

    if best_cluster_info is None:
        return {"result": "not selected", "statistics": cluster_statistics, "selected_cluster_id": None}

    chosen_indices = best_cluster_info["indices"]
    best_idx = max(chosen_indices, key=lambda idx: len(cand_list[idx]["sources"]))

    result = cand_list[best_idx].copy()
    result.update({
        "short_answer": best_cluster_info["label"],
        "selection_status": "selected",
        "cluster_score": round(best_cluster_info["cluster_score"], 4),
        "total_candidates_in_cluster": len(chosen_indices)
    })

    return {
        "result": result,
        "statistics": cluster_statistics,
        "selected_cluster_id": selected_cluster_id
    }


def lcs_length(s1: str, s2: str) -> int:
    """
    Tính độ dài chuỗi con chung dài nhất (LCS) giữa s1 và s2.
    Sử dụng 2 hàng để tối ưu bộ nhớ.
    """
    s1, s2 = str(s1).lower().strip(), str(s2).lower().strip()
    n, m = len(s1), len(s2)
    
    if n == 0 or m == 0:
        return 0
        
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[m]

def cluster_and_selected_candidate_WH(golden_answer, cand_list, model, embedding_cache, MIN_SCORE_THRESHOLD=2, DISTANCE_THRESHOLD=0.12):
    if not cand_list:
        return {"result": "not selected", "statistics": {}, "selected_cluster_id": None}

    # --- 1. ENCODING & CLUSTERING (Giữ nguyên) ---
    texts = [c["text"] for c in cand_list]
    unique_texts = list(set(texts))
    missing_texts = [t for t in unique_texts if t not in embedding_cache]
    if missing_texts:
        new_embeddings = model.encode(missing_texts)
        for t, emb in zip(missing_texts, new_embeddings):
            embedding_cache[t] = emb
    embeddings = np.array([embedding_cache[t] for t in texts])
    
    cluster_labels = []
    if len(cand_list) >= 2:
        cluster_labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=DISTANCE_THRESHOLD,
            metric="cosine", linkage="average"
        ).fit(embeddings).labels_
    else:
        cluster_labels = [0]

    cluster_statistics = {}
    raw_clusters = []
    unique_labels = set(cluster_labels)

    for lbl in unique_labels:
        indices = [i for i, l in enumerate(cluster_labels) if l == lbl]
        individual_ref_counts = [len(cand_list[i].get("sources", [])) for i in indices]
        
        ct1 = calculate_gini_adjusted_score(individual_ref_counts)
        ct2 = score_v2_log(individual_ref_counts)
        ct3 = score_v3_sigmoid(individual_ref_counts)

        cluster_id = f"cluster_{lbl}"
        cluster_statistics[cluster_id] = {
            "num_candidates": len(indices),
            "candidate_evidences": {idx: len(cand_list[idx].get("sources", [])) for idx in indices},
            "scores": {
                "ct_1": round(ct1, 4),
                "ct_2": round(ct2, 4),
                "ct_3": round(ct3, 4)
            }
        }

        raw_clusters.append({
            "cluster_id": cluster_id,
            "indices": indices,
            "cluster_score": ct1
        })

    # --- 2. LỌC CLUSTER ---
    best_cluster_info = None
    selected_cluster_id = None
    if raw_clusters:
        candidate_best = max(raw_clusters, key=lambda x: x["cluster_score"])
        if candidate_best["cluster_score"] >= MIN_SCORE_THRESHOLD:
            best_cluster_info = candidate_best
            selected_cluster_id = candidate_best["cluster_id"]

    if best_cluster_info is None:
        return {"result": "not selected", "statistics": cluster_statistics, "selected_cluster_id": None}
    
    # --- 3. MERGING & LONG ANSWER (Giữ nguyên logic của bạn) ---
    chosen_indices = best_cluster_info["indices"]
    merged_in_cluster = {}
    for idx in chosen_indices:
        cand = cand_list[idx]
        norm_text = cand["text"]
        if norm_text not in merged_in_cluster:
            merged_in_cluster[norm_text] = {
                "text": cand["text"],
                "sources": set(cand.get("sources", [])),
                "long_answer": cand.get('long_answer', ""),
                'list_cands': [idx]
            }
        else:
            merged_in_cluster[norm_text]["sources"].update(cand.get("sources", []))
            merged_in_cluster[norm_text]['list_cands'].append(idx)

    result_group = max(merged_in_cluster.values(), key=lambda x: len(x["sources"]))
    
    best_cand_idx_in_group = -1
    max_lcs_val = -1
    for idx in result_group['list_cands']:
        current_lcs = lcs_length(cand_list[idx].get('raw_ans', cand_list[idx]['text']), golden_answer)
        if current_lcs > max_lcs_val:
            max_lcs_val = current_lcs
            best_cand_idx_in_group = idx
    
    winner_cand = cand_list[best_cand_idx_in_group]

    final_res = {
        "text": result_group["text"],
        "long_answer": winner_cand["long_answer"], 
        "sources": list(result_group["sources"]),  
        "selection_status": "selected",
        "cluster_score": round(best_cluster_info["cluster_score"], 4)
    }

    return {
        "result": final_res,
        "statistics": cluster_statistics,
        "selected_cluster_id": selected_cluster_id
    }
    
def process_get_final_answer(golden_answer, target_answer_type, cluster_raw, mapping_data, embedding_model, embedding_cache):
    candidates = collect_candidates(cluster_raw, mapping_data) 
    if target_answer_type != 'correctness':
        final_output = cluster_and_selected_candidate_WH(golden_answer, candidates, embedding_model, embedding_cache) 
    else:
        final_output = cluster_and_selected_candidate_YN(candidates)
        
    if final_output == 'not selected':
        return {
                "short_answer": "HELLO THU",
                "long_answer": "HELLO THU",
                "selection_status": "zero-shot_fallback",
                "cluster_score": None
            }
        
    return final_output

    