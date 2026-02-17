import json
import re
import numpy as np
import os
import sys
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from postprocessing import postprocessing

DISTANCE_THRESHOLD = 0.12
USE_SOURCE_BONUS = False 
SOURCE_DISTANCE_MARGIN = 0.05
GG_BONUS = 0.2
WIKI_BONUS = 0.3

def load_json(path):
    if not os.path.exists(path):
        print(f"Lỗi: Không tìm thấy file tại {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_references(long_answer):
    if not long_answer:
        return []
    return re.findall(r"\[([^\]]+)\]", long_answer)

def extract_source(ref):
    parts = ref.split("_")
    source_parts = []
    for p in parts:
        if p.isdigit():
            break
        source_parts.append(p)
    return "_".join(source_parts)

def cluster_cohesion(embeddings):
    if len(embeddings) <= 1:
        return 0
    centroid = embeddings.mean(axis=0)
    dists = np.linalg.norm(embeddings - centroid, axis=1)
    return dists.mean()

def collect_candidates(cluster_file, mapping_data):
    """
    Đọc file JSON gốc, lọc và chuẩn hóa dữ liệu ứng viên.
    """
    raw_data = load_json(cluster_file)
    processed_data = {}

    for qidx, clusters in raw_data.items():
        cand_list = []
        for cid, cluster in clusters.items():
            if "llm" not in cluster:
                continue

            ans = cluster["llm"].get("short_answer", "").strip().lower()
            long_answer = cluster["llm"].get("long_answer", "")
            context = cluster.get("context", "").lower()
            
            if not ans or ans == "insufficient information":
                continue

            # Chuẩn hóa (Normalization)
            if ans in mapping_data:
                norm_ans = mapping_data[ans]
            else:
                norm_ans = postprocessing(ans)
                norm_ans = mapping_data.get(norm_ans, norm_ans)

            # Thu thập nguồn ngay từ bước này để tối ưu tốc độ
            refs = extract_references(long_answer)
            sources = [extract_source(r) for r in refs]
            has_wiki = any("WIKI" in s for s in sources)
            has_gg = any("GG" in s for s in sources)

            cand_list.append({
                "text": norm_ans,
                "long_answer": long_answer,
                "context": context,
                "sources": sources,
                "has_wiki": has_wiki,
                "has_gg": has_gg
            })
        
        # Nếu có ít hơn 2 ứng viên, coi như không đủ dữ liệu để fusion
        processed_data[qidx] = cand_list if len(cand_list) >= 2 else None
            
    return processed_data

# =========================
# 2. CLUSTER & EVALUATE FUNCTION
# =========================
def cluster_and_evaluate(processed_data, model):
    """
    Thực hiện phân cụm và chọn ra câu trả lời cuối cùng.
    """
    fused_results = {}

    for qidx, cand_list in processed_data.items():
        if cand_list is None:
            fused_results[qidx] = {"abstain": True}
            continue

        texts = [c["text"] for c in cand_list]
        embeddings = model.encode(texts)

        # Thuật toán phân cụm
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=DISTANCE_THRESHOLD,
            metric="cosine",
            linkage="average"
        ).fit(embeddings).labels_

        best_cluster = None
        best_cohesion = 1e9
        raw_clusters = []

        for lbl in set(labels):
            indices = [i for i, l in enumerate(labels) if l == lbl]
            evidence_count = len(indices)
            
            # Tính toán các thuộc tính của cụm
            cluster_sources = set()
            has_wiki = False
            has_gg = False
            for i in indices:
                cluster_sources.update(cand_list[i]["sources"])
                if cand_list[i]["has_wiki"]: has_wiki = True
                if cand_list[i]["has_gg"]: has_gg = True

            unique_source_count = len(cluster_sources)
            raw_clusters.append({
                "indices": indices,
                "evidence_count": evidence_count,
                "has_wiki": has_wiki,
                "has_gg": has_gg,
                "unique_source": unique_source_count
            })

            # RULE: Ưu tiên cụm có đông đảo sự đồng thuận (size >= 3 hoặc từ nhiều nguồn khác nhau)
            if not (evidence_count >= 3 or unique_source_count >= 4):
                continue

            cohesion = cluster_cohesion(embeddings[indices])
            if cohesion < best_cohesion:
                best_cluster = indices
                best_cohesion = cohesion

        # FALLBACK RULE
        if best_cluster is None and len(raw_clusters) == 2:
            c1, c2 = raw_clusters
            if c1["evidence_count"] != c2["evidence_count"]:
                larger = c1 if c1["evidence_count"] > c2["evidence_count"] else c2
                if larger["has_wiki"]:
                    best_cluster = larger["indices"]

        # SINGLE CLUSTER RULE
        if best_cluster is None and len(raw_clusters) == 1:
            only_cluster = raw_clusters[0]
            if only_cluster["evidence_count"] == 2 and only_cluster["has_wiki"]:
                best_cluster = only_cluster["indices"]

        if best_cluster is None:
            fused_results[qidx] = {"abstain": True}
            continue

        # CHỌN FINAL ANSWER TRONG CỤM TỐT NHẤT
        centroid = embeddings[best_cluster].mean(axis=0)
        distances = {i: np.linalg.norm(embeddings[i] - centroid) for i in best_cluster}
        min_dist = min(distances.values())

        def candidate_score(idx):
            dist = distances[idx]
            if not USE_SOURCE_BONUS: return dist
            if dist - min_dist > SOURCE_DISTANCE_MARGIN: return dist
            if cand_list[idx]["has_gg"]: return dist - GG_BONUS
            if cand_list[idx]["has_wiki"]: return dist - WIKI_BONUS
            return dist

        best_idx = min(best_cluster, key=candidate_score)

        fused_results[qidx] = {
            "final_answer": cand_list[best_idx]["text"],
            "context": cand_list[best_idx]["context"],
            "abstain": False
        }

    return fused_results

# =========================
# EVALUATION SUMMARY
# =========================
def compute_selected_summary(fused_results, dict_golden):
    total_with_evidence = len(fused_results)
    selected, correct_selected, total_wrong = 0, 0, 0
    wrong_has_golden_in_ctx, wrong_no_golden_in_ctx = 0, 0

    for qidx, info in fused_results.items():
        if info.get("abstain", True):
            continue

        selected += 1
        pred = str(info.get("final_answer", "")).lower().strip()
        context = str(info.get("context", "")).lower()
        golden_list = [str(g).lower().strip() for g in dict_golden.get(qidx, {}).get("True", [])]

        if pred in golden_list:
            correct_selected += 1
        else:
            total_wrong += 1
            if any(g in context for g in golden_list):
                wrong_has_golden_in_ctx += 1
            else:
                wrong_no_golden_in_ctx += 1

    acc = correct_selected / selected if selected else 0.0
    coverage = selected / total_with_evidence if total_with_evidence else 0.0

    return {
        "total": total_with_evidence, "selected": selected, "correct": correct_selected,
        "acc": acc, "coverage": coverage, "wrong": total_wrong,
        "wrong_in_ctx": wrong_has_golden_in_ctx, "wrong_no_ctx": wrong_no_golden_in_ctx
    }

# =========================
# MAIN EXECUTION
# =========================
def load_pubmed_model():
    print("Loading embedding model (PubMedBERT)...")
    static = StaticEmbedding.from_model2vec("neuml/pubmedbert-base-embeddings-8M")
    return SentenceTransformer(modules=[static])

if __name__ == "__main__":
    # --- Cấu hình đường dẫn ---
    s_type = "kmean"
    n_cluster = 7
    PATH_CLUSTER = fr"src\cluster\final\concat_answer_{s_type}_merge_google_wiki_{n_cluster}.json"
    PATH_GOLDEN = "data/all_answer_golden_56.json"
    PATH_MAPPING = 'data/mapping_answer_gol.json'
    PATH_OUTPUT = r"src\cluster\final\fused_results.json"

    # --- Thực thi ---
    model = load_pubmed_model()
    mapping_data = load_json(PATH_MAPPING)
    dict_golden = load_json(PATH_GOLDEN)

    print("Step 1: Collecting candidates...")
    processed_candidates = collect_candidates(PATH_CLUSTER, mapping_data)

    print("Step 2: Running clustering and fusion logic...")
    results = cluster_and_evaluate(processed_candidates, model)

    print(f"Step 3: Saving results to {PATH_OUTPUT}...")
    with open(PATH_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # --- Thống kê ---
    stats = compute_selected_summary(results, dict_golden)
    print("\n" + "="*30)
    print("STATISTICS REPORT")
    print("="*30)
    print(f"Total Questions    : {stats['total']}")
    print(f"Selected Answers   : {stats['selected']} (Coverage: {stats['coverage']*100:.2f}%)")
    print(f"Correct Answers    : {stats['correct']} (Accuracy: {stats['acc']*100:.2f}%)")
    print(f"Total Errors       : {stats['wrong']}")
    print(f"  - Golden in Ctx  : {stats['wrong_in_ctx']}")
    print(f"  - Golden NOT in Ctx: {stats['wrong_no_ctx']}")
    print("="*30)