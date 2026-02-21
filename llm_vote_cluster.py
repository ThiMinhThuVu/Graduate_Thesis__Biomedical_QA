import json
import re
import numpy as np
import os
import math
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from postprocessing import postprocessing
from src.gpt_request import request_api


DISTANCE_THRESHOLD = 0.12
BASE_FOLDER = "src/cluster/final"
PATH_GOLDEN = "data/all_answer_golden_56.json"
PATH_MAPPING = 'data/mapping_answer_gol.json'
PATH_METADATA = 'data/metadata_medhop.json'
PATH_CLUSTER = r'src\cluster\final\WH_answer_kmean_merge_google_wiki_7_only_GG.json'


def load_json(path):
    if not os.path.exists(path):
        print(f"Lỗi: Không tìm thấy file tại {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def extract_references(long_answer):
    if not long_answer:
        return []
    return re.findall(r"\[([^\]]+)\]", long_answer)

def load_pubmed_model():
    print("Loading embedding model (PubMedBERT)...")    
    return SentenceTransformer("pritamdeka/S-BioBert-snli-multinli-stsb", device="cpu")

def collect_candidates(cluster_file, mapping_data):
    raw_data = load_json(cluster_file)
    processed_data = {}

    for qidx, clusters in raw_data.items():
        cand_list = []
        for cid, cluster in clusters.items():
            if "llm" not in cluster:
                continue

            ans = cluster["llm"].get("short_answer", "").strip().lower()
            long_answer = cluster["llm"].get("long_answer", "")
            context = cluster.get("context", "")
            
            if not ans or ans == "insufficient information":
                continue

            # Normalization
            if ans in mapping_data:
                norm_ans = mapping_data[ans]
            else:
                norm_ans = postprocessing(ans)
                norm_ans = mapping_data.get(norm_ans, norm_ans)
                        
            sources = list(set(extract_references(long_answer)))
   
            cand_list.append({
                "cid": cid,
                "text": norm_ans,
                "long_answer": long_answer,
                "sources": sources,
                "context": context  
            })
        
        processed_data[qidx] = cand_list if len(cand_list) >= 1 else None
        
    return processed_data

# ==========================================
# LLM CLUSTER SELECTION
# ==========================================
def select_best_cluster_via_llm(question_text, raw_clusters, cand_list):
    if not raw_clusters:
        return None, "No clusters"

    clusters_description = ""
    for i, cluster in enumerate(raw_clusters):
        candidate_details = []
        for idx in cluster["indices"]:
            cand = cand_list[idx]
            detail = f"   + Short Answer: {cand['text']}\n"
            detail += f"     Supporting Context: {cand['long_answer']}" 
            candidate_details.append(detail)
        
        candidates_text = "\n".join(candidate_details)        
        clusters_description += f"--- CLUSTER ID: {i} ---\n"
        clusters_description += f"List of Evidence Pairs:\n{candidates_text}\n\n"

    prompt = f"""You are an expert Biomedical Consensus Aggregator. Your objective is to choose a cluster from multiple retrieved cluster answers for a specific informational query.

        ### INPUT DATA:
        - **Question**: "{question_text}"
        - **Cluster Answers**: {clusters_description}: Each cluster consists of many candidates, each candidate has a short answer, a long answer, and the provided context.

        ### CRITICAL EVALUATION RULES:
        1.  **EVIDENCE & CONFIDENCE EXTRACTION:**
        -   **High Confidence > Low Confidence**: Prioritize clusters where candidates have long answers containing strong affirmation keywords (e.g., "explicitly", "clearly", …) over tentative language (e.g., "suggests", "might", "potentially", …).
        2.  **ASPECT COVERAGE:** Prioritize clusters that have covered every aspect of the question.
        3. **NO MAJORITY VOTING & STRICT VALIDATION:** Cluster with more candidates does not ensure correction

        ### DECISION PROCESS:
        **Step 1: Extract & Evaluate**
        Analyze the question and each cluster.
        Ranking clusters based on confidence and aspect coverage
        Choose the highest rank cluster.
        **Step 2: Draft Long Answer**
        - Synthesize a comprehensive explanation why that cluster is chosen.

        ### OUTPUT FORMAT:
        Provide the output strictly as a valid JSON object. Do not include any introductory or concluding text outside the JSON block.
        {{
        "best_cluster_index": <int>,
        "reason": "<Explain why this cluster's [Short + Long] pairs are the most medically sound>"
        }}"""


    try:
        response_str = request_api(prompt, model='o3-mini') 
        match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if match:
            res_data = json.loads(match.group())
            idx = res_data.get("best_cluster_index")
            if 0 <= idx < len(raw_clusters):
                selected = raw_clusters[idx]
                selected["llm_reason"] = res_data.get("reason", "")
                selected["cluster_index"] = idx
                return selected, "llm_decision"
    except Exception as e:
        print(f"Lỗi khi LLM chọn cụm cho câu hỏi: {e}")
        
    return None, "selection_failed_or_error"

# ==========================================
# CORE EVALUATION LOGIC
# ==========================================
def cluster_and_evaluate(metadata, processed_data, model, dict_golden):
    fused_results = {}
    list_id_miss = set()
    detailed_logs = {}

    for qidx, cand_list in processed_data.items():
        if not cand_list or len(cand_list) < 2:
            fused_results[qidx] = {"abstain": True}
            list_id_miss.add(qidx)
            continue
            
        question_text = metadata[qidx]['original question']
        golden_answers = [str(g).lower().strip() for g in dict_golden[qidx]['True']]
        
        # --- BƯỚC 1: PHÂN CỤM ---
        texts = [c["text"] for c in cand_list]
        embeddings = model.encode(texts)
        
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=DISTANCE_THRESHOLD, 
            metric="cosine", linkage="average"
        ).fit(embeddings).labels_

        raw_clusters = []
        for lbl in set(labels):
            indices = [i for i, l in enumerate(labels) if l == lbl]
            total_refs = sum([len(cand_list[i]["sources"]) for i in indices])
            raw_clusters.append({
                "indices": indices,
                "evidence_count": len(indices),
                "unique_source": total_refs
            })
        
        # --- BƯỚC 2: LLM CHỌN CỤM ---
        best_cluster, method = select_best_cluster_via_llm(question_text, raw_clusters, cand_list)

        # --- BƯỚC 3: KIỂM TRA "HIT" GOLDEN TRONG CỤM ---
        if best_cluster:
            chosen_indices = best_cluster["indices"]
            cluster_contains_golden = False
            matched_candidate = None
            
            for idx in chosen_indices:
                cand_text = str(cand_list[idx]["text"]).lower().strip()
                if cand_text in golden_answers:
                    cluster_contains_golden = True
                    matched_candidate = cand_list[idx]["text"]
                    break 

            fused_results[qidx] = {
                "final_answer": None, # Tạm để None theo yêu cầu
                "abstain": False,
                "is_correct_cluster": cluster_contains_golden,
                "debug_info": {
                    "method": method,
                    "reason": best_cluster.get("llm_reason"),
                    "hit_golden": cluster_contains_golden
                }
            }

            detailed_logs[qidx] = {
                "question": question_text,
                "selected_cluster_idx": best_cluster.get("cluster_index"),
                "llm_reason": best_cluster.get("llm_reason"),
                "is_correct_cluster": cluster_contains_golden,
                "matched_with_golden_text": matched_candidate,
                "all_candidates_in_cluster": [cand_list[i]["text"] for i in chosen_indices]
            }
        else:
            fused_results[qidx] = {"abstain": True}
            list_id_miss.add(qidx)

    save_json(detailed_logs, 'cache_logs/full_cluster_llm_logs.json')
    return fused_results, list(list_id_miss)

# ==========================================
# SUMMARY STATS
# ==========================================
def compute_selected_summary(fused_results):
    total_processed = len(fused_results)
    selected, correct_clusters = 0, 0
    list_id_wrong = []

    for qidx, info in fused_results.items():
        if info.get("abstain", True):
            continue

        selected += 1
        if info.get("is_correct_cluster", False):
            correct_clusters += 1
        else:
            list_id_wrong.append(qidx)

    acc = correct_clusters / selected if selected else 0.0
    coverage = selected / total_processed if total_processed else 0.0

    return {
        "total": total_processed, "selected": selected, "correct_clusters": correct_clusters,
        "acc_cluster": acc, "coverage": coverage
    }, list_id_wrong


if __name__ == "__main__":
    # Load data
    model = load_pubmed_model()
    mapping_data = load_json(PATH_MAPPING)
    dict_golden = load_json(PATH_GOLDEN)
    metadata = load_json(PATH_METADATA)

    print("Step 1: Collecting candidates...")
    processed_candidates = collect_candidates(PATH_CLUSTER, mapping_data)

    print("Step 2: Clustering and LLM Evaluation...")
    results, list_id_miss = cluster_and_evaluate(metadata, processed_candidates, model, dict_golden)

    print("Step 3: Computing Results...")
    save_json(list_id_miss, "list_id_miss_llm.json")
    stats, list_id_wrong = compute_selected_summary(results)

    print("-" * 30)
    print(f"Total Questions    : {stats['total']}")
    print(f"Selected (Coverage): {stats['selected']} ({stats['coverage']*100:.2f}%)")
    print(f"Correct Clusters   : {stats['correct_clusters']} (Acc: {stats['acc_cluster']*100:.2f}%)")
    print("-" * 30)