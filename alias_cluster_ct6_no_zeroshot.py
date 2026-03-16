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

def load_embedding_cache(path, timeout=60):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return {}

    try:
        with portalocker.Lock(path, mode='r', timeout=timeout, encoding='utf-8') as f:
            data = json.load(f)
            # Convert list ngược lại thành numpy array
            return {k: np.array(v) for k, v in data.items()}
    except Exception as e:
        print(f"⚠️ Cache embedding lỗi hoặc trống: {e}")
        return {}

def save_embedding_cache(new_data, path, timeout=60):
    if not new_data:
        return

    try:
        with portalocker.Lock(path, mode='r+', timeout=timeout, encoding='utf-8') as f:
            # Đọc cache hiện tại
            try:
                f.seek(0)
                current_cache = json.load(f)
            except:
                current_cache = {}

            # Cập nhật dữ liệu mới (chỉ convert numpy sang list tại đây)
            for text, emb in new_data.items():
                if isinstance(emb, np.ndarray):
                    current_cache[text] = emb.tolist()
                else:
                    current_cache[text] = emb

            # Ghi lại
            f.seek(0)
            f.truncate()
            json.dump(current_cache, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Lỗi ghi embedding cache: {e}")
        
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
        return {"selection_status": "not selected", "stats": {"num_clusters": 0, "cluster_sizes": [], "best_cluster_idx": None}}

    clusters = defaultdict(list)
    for i, c in enumerate(cand_list):
        label = c["text"].strip().lower()
        clusters[label].append(i)

    cluster_labels = list(clusters.keys())
    cluster_sizes = [len(clusters[lbl]) for lbl in cluster_labels]
    
    raw_clusters = []
    for i, label in enumerate(cluster_labels):
        indices = clusters[label]
        individual_ref_counts = [len(cand_list[idx].get("sources", [])) for idx in indices]
        c_score = calculate_gini_adjusted_score(individual_ref_counts)
        raw_clusters.append({
            "label": label, 
            "indices": indices, 
            "cluster_score": c_score,
            "internal_idx": i
        })

    best_cluster_info = None
    best_cluster_idx = None
    if raw_clusters:
        candidate_best = max(raw_clusters, key=lambda x: x["cluster_score"])
        if candidate_best["cluster_score"] >= MIN_SCORE_THRESHOLD:
            best_cluster_info = candidate_best
            best_cluster_idx = candidate_best["internal_idx"]

    stats_info = {
        "num_clusters": len(clusters), 
        "cluster_sizes": cluster_sizes,
        "best_cluster_idx": best_cluster_idx
    }

    if best_cluster_info is None:
        return {"selection_status": "not selected", "stats": stats_info}

    # Merge logic tương tự WH
    chosen_indices = best_cluster_info["indices"]
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
            merged_in_cluster[norm_text]["long_answer"] += f" {cand.get('long_answer', '')}"

    # Thống kê số lượng unique sau khi merge
    stats_info["num_unique_in_best_cluster"] = len(merged_in_cluster)
    
    # Chọn ứng viên có nhiều nguồn nhất trong cụm thắng cuộc
    result = max(merged_in_cluster.values(), key=lambda x: len(x["sources"]))
    result["sources"] = list(result["sources"])
    result.update({
        "short_answer": best_cluster_info["label"],
        "selection_status": "selected",
        "cluster_score": round(best_cluster_info["cluster_score"], 4),
        "stats": stats_info
    })
    return result

def cluster_and_selected_candidate_WH(cand_list, model, MIN_SCORE_THRESHOLD=2, DISTANCE_THRESHOLD=0.12, global_cache={}):
    if not cand_list:
        return {"selection_status": "not selected", "stats": {"num_clusters": 0, "cluster_sizes": [], "best_cluster_idx": None}}

    # ... (giữ nguyên phần encode embedding) ...
    texts = [c["text"] for c in cand_list]
    unique_texts = list(set(texts))
    missing_texts = [t for t in unique_texts if t not in global_cache]
    if missing_texts:
        new_embeddings = model.encode(missing_texts)
        global_cache.update(dict(zip(missing_texts, new_embeddings)))
    embeddings = np.array([global_cache[t] for t in texts])
    
    cluster_sizes = []
    raw_clusters = []

    if len(cand_list) >= 2:
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=DISTANCE_THRESHOLD,
            metric="cosine", linkage="average"
        ).fit(embeddings).labels_

        unique_labels = sorted(list(set(labels)))
        for i, lbl in enumerate(unique_labels):
            indices = [idx for idx, l in enumerate(labels) if l == lbl]
            cluster_sizes.append(len(indices))
            
            individual_ref_counts = [len(cand_list[idx]["sources"]) for idx in indices]
            c_score = calculate_gini_adjusted_score(individual_ref_counts)
            raw_clusters.append({"indices": indices, "cluster_score": c_score, "internal_idx": i})
    else:
        cluster_sizes = [1]
        raw_clusters.append({"indices": [0], "cluster_score": calculate_gini_adjusted_score([len(cand_list[0].get("sources", []))]), "internal_idx": 0})

    best_cluster_info = None
    best_cluster_idx = None
    if raw_clusters:
        candidate_best = max(raw_clusters, key=lambda x: x["cluster_score"])
        if candidate_best["cluster_score"] >= MIN_SCORE_THRESHOLD:
            best_cluster_info = candidate_best
            best_cluster_idx = candidate_best["internal_idx"]

    stats_info = {
        "num_clusters": len(cluster_sizes), 
        "cluster_sizes": cluster_sizes,
        "best_cluster_idx": best_cluster_idx
    }

    if best_cluster_info is None:
        return {"selection_status": "not selected", "stats": stats_info}
    
    # Merge logic cho các alias trong cluster thắng cuộc
    chosen_indices = best_cluster_info["indices"]
    merged_in_cluster = {}
    for idx in chosen_indices:
        cand = cand_list[idx]
        norm_text = cand["text"]
        if norm_text not in merged_in_cluster:
            merged_in_cluster[norm_text] = {
                "text": cand["text"], "sources": set(cand.get("sources", [])),
                "context": cand.get("context", ""), "long_answer": cand.get('long_answer', "")
            }
        else:
            merged_in_cluster[norm_text]["sources"].update(cand.get("sources", []))
            merged_in_cluster[norm_text]["long_answer"] += f" {cand['long_answer']}"

    # --- PHẦN CHỈNH SỬA: Thống kê số lượng sau khi merge ---
    num_unique_candidates_in_best_cluster = len(merged_in_cluster)
    stats_info["num_unique_in_best_cluster"] = num_unique_candidates_in_best_cluster
    # ------------------------------------------------------
    
    result = max(merged_in_cluster.values(), key=lambda x: len(x["sources"]))
    result["sources"] = list(result["sources"])
    result.update({
        "short_answer": result["text"],
        "selection_status": "selected",
        "cluster_score": round(best_cluster_info["cluster_score"], 4),
        "stats": stats_info
    })
    return result

def process_get_final_answer(target_answer_type, cluster_raw, mapping_data, embedding_model, global_cache):
    candidates = collect_candidates(cluster_raw, mapping_data) 
    
    if target_answer_type != 'correctness':
        final_output = cluster_and_selected_candidate_WH(candidates, embedding_model, global_cache=global_cache) 
    else:
        final_output = cluster_and_selected_candidate_YN(candidates)
        
    if final_output["selection_status"] != "selected":
        # Đảm bảo fallback vẫn giữ stats để thống kê không bị thiếu qidx
        final_output.update({
            "short_answer": "HELLO THU",
            "long_answer": "HELLO THU",
            "selection_status": "zero-shot_fallback",
            "cluster_score": None
        })
        
    return final_output