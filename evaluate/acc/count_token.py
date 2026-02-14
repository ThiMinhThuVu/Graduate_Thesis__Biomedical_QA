import os
import json
import csv
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from postprocessing import postprocessing

# =========================
# CONFIG
# =========================

SOURCE_FAMILY = {
    "google": "search",
    "wiki": "wiki",
    "crawl_wiki": "wiki",
    "crawl_ner_wiki": "wiki",
    "crawl_sparql_wikidata": "kg",
    "create_MCQ": "dense"
}

DISTANCE_THRESHOLD = 0.12
GOLDEN_PATH = "data/all_answer_golden_56.json"
CSV_OUT = "evaluate/acc/fusion_trace.csv"

# =========================
# LOAD DATA
# =========================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

mapping_data = load_json("data/mapping_answer_gol.json")
dict_golden = load_json(GOLDEN_PATH)

# =========================
# MODEL
# =========================

def load_pubmed_model():
    static = StaticEmbedding.from_model2vec(
        "neuml/pubmedbert-base-embeddings-8M"
    )
    return SentenceTransformer(modules=[static])

# =========================
# UTILS
# =========================

def detect_source_flags(cluster_sources):
    families = {SOURCE_FAMILY.get(src, src) for src, _ in cluster_sources}
    return (
        "dense" in families,
        "wiki" in families,
        "search" in families,
        families
    )

def cluster_cohesion(embeds):
    centroid = embeds.mean(axis=0, keepdims=True)
    sim = cosine_similarity(embeds, centroid).reshape(-1)
    return float(np.mean(1 - sim))

# =========================
# COLLECT CANDIDATES
# =========================

def collect_wh_candidates(base_paths, dict_golden, run_type=None):
    candidates = []
    qidx_meta = {}

    for path in base_paths:
        all_ans_path = os.path.join(path, "all_ans")
        if not os.path.exists(all_ans_path):
            continue

        for fname in os.listdir(all_ans_path):
            if not fname.endswith(".json"):
                continue

            with open(os.path.join(all_ans_path, fname), "r", encoding="utf-8") as f:
                data = json.load(f)

            qidx = data.get("QIDX")
            question = data.get("original_question", "").strip()

            if qidx not in dict_golden:
                continue

            if data["type_q"] != "direct" or data["target_answer_type"] == "correctness":
                continue

            if qidx not in qidx_meta:
                golden_list = dict_golden[qidx].get("True", [])
                qidx_meta[qidx] = {
                    "golden": golden_list,
                    "question": question,
                    "has_golden": len(golden_list) > 0
                }

            if not qidx_meta[qidx]["has_golden"]:
                continue

            # -------- Dense --------
            if run_type == "DENSE":
                for cand in data.get("create_mcq", []):
                    ans = cand.get("short_answer", "").strip().lower()
                    if not ans or ans == "insufficient information":
                        continue

                    ans = mapping_data.get(ans, mapping_data.get(postprocessing(ans), postprocessing(ans)))

                    candidates.append({
                        "qidx": qidx,
                        "text": ans,
                        "long_answer": cand.get("long_answer", "").strip(),
                        "source": "create_MCQ",
                        "run": run_type
                    })

            # -------- Sparse --------
            else:
                source_outputs = (
                    data.get("1", {})
                        .get("0", {})
                        .get("0", {})
                        .get("source_output", {})
                )

                for src, res in source_outputs.items():
                    for i_c, cand in enumerate(res.get("candidate", [])):
                        ans = cand.get("short_answer", "").lower()
                        if not ans or ans == "insufficient information":
                            continue

                        ans = mapping_data.get(ans, mapping_data.get(postprocessing(ans), postprocessing(ans)))

                        candidates.append({
                            "qidx": qidx,
                            "text": ans,
                            "long_answer": cand.get("long_answer", "").strip(),
                            "source": src,
                            "run": f"{run_type}_{i_c}"
                        })

    return candidates, qidx_meta

# =========================
# CLUSTER + EVALUATE
# =========================

def cluster_and_evaluate(candidates, qidx_meta, model):
    grouped = defaultdict(list)
    for c in candidates:
        grouped[c["qidx"]].append(c)

    fused_results = {}

    for qidx, cand_list in grouped.items():

        if len(cand_list) == 1:
            final_ans = cand_list[0]["text"]
            norm_ans = mapping_data.get(final_ans, mapping_data.get(postprocessing(final_ans), postprocessing(final_ans)))

            fused_results[qidx] = {
                "final_answer": final_ans,
                "norm_ans": norm_ans,
                "is_correct_norm": norm_ans in qidx_meta[qidx]["golden"],
                "abstain": False
            }
            continue

        texts = [c["text"] for c in cand_list]
        embeddings = model.encode(texts)

        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=DISTANCE_THRESHOLD,
            metric="cosine",
            linkage="average"
        ).fit(embeddings).labels_

        cluster_info = []

        for lbl in set(labels):
            indices = [i for i, l in enumerate(labels) if l == lbl]
            cluster_sources = {(cand_list[i]["source"], cand_list[i]["run"]) for i in indices}
            has_dense, has_wiki, _, families = detect_source_flags(cluster_sources)

            if not (len(indices) >= 3 and has_dense and has_wiki):
                continue

            cohesion = cluster_cohesion(embeddings[indices])

            cluster_info.append({
                "indices": indices,
                "cohesion": cohesion
            })

        if not cluster_info:
            fused_results[qidx] = {"abstain": True}
            continue

        best = min(cluster_info, key=lambda x: x["cohesion"])
        idx = best["indices"][0]

        final_ans = cand_list[idx]["text"]
        norm_ans = mapping_data.get(final_ans, mapping_data.get(postprocessing(final_ans), postprocessing(final_ans)))

        fused_results[qidx] = {
            "final_answer": final_ans,
            "norm_ans": norm_ans,
            "is_correct_norm": norm_ans in qidx_meta[qidx]["golden"],
            "abstain": False
        }

    return {"details": fused_results}

# =========================
# BUILD WRONG MCQ
# =========================

def build_wrong_answer_mcq_candidates(stats, all_candidates):
    cand_by_qidx = defaultdict(list)
    for c in all_candidates:
        cand_by_qidx[c["qidx"]].append(c)

    mcq_pool = {}

    for qidx, info in stats["details"].items():
        # Chỉ xét các qidx mà kết quả là abstain (không chọn đáp án nào theo rule)
        if not info.get("abstain", False):
            continue

        # 🔥 lấy evidence thực tế
        evidence_list = cand_by_qidx.get(qidx, [])

        # ❌ loại no-evidence
        if len(evidence_list) == 0:
            continue

        # ✅ câu này: có evidence nhưng không được chọn làm đáp án cuối cùng
        bucket = defaultdict(list)

        for c in evidence_list:
            ans = c["text"]
            if ans in  mapping_data:
                norm_ans = mapping_data[ans]
            else:
                norm_ans = postprocessing(ans)
                norm_ans = mapping_data.get(norm_ans, norm_ans)
            # Bỏ qua nếu norm_ans là 'nan' (không hợp lệ)
            if norm_ans == "nan":
                continue
            long_ans = c.get("long_answer", "").strip()
            if long_ans:
                bucket[norm_ans].append(long_ans)

        merged = []
        for ans, reasons in bucket.items():
            merged.append({
                "short_answer": ans,
                "num_evidence": len(reasons),
                "final_text": f"{ans}. Reason: " + " | ".join(reasons)
            })

        if merged and len(merged) > 1:
            mcq_pool[qidx] = merged

    return mcq_pool

# =========================
# TOKEN ESTIMATION
# =========================

def estimate_token_count(mcq_pool):
    text = ""
    for cands in mcq_pool.values():
        for c in cands:
            text += c["final_text"] + "\n"

    words = len(text.split())
    tokens = int(words * 1.3)

    return words, tokens

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    model = load_pubmed_model()

    c1, meta = collect_wh_candidates(
        ["medhop_results_mcq_2_TFIDF1000"],
        dict_golden,
        "TFIDF1000"
    )
    c2, _ = collect_wh_candidates(
        ["medhop_results_mcq_3_TFIDF300"],
        dict_golden,
        "TFIDF300"
    )
    c3, _ = collect_wh_candidates(
        ["medhop_results_no_mcq_TFIDF300"],
        dict_golden,
        "DENSE"
    )

    all_candidates = c1 + c2 + c3

    stats = cluster_and_evaluate(all_candidates, meta, model)

    mcq_candidates = build_wrong_answer_mcq_candidates(stats, all_candidates)

    with open("mcq_candidate_wh.json", 'w', encoding='utf-8') as f:
        json.dump(mcq_candidates, f, indent=2, ensure_ascii=False)
        
    words, tokens = estimate_token_count(mcq_candidates)

    print("\n=== WRONG ANSWER MCQ POOL ===")
    print(f"Questions : {len(mcq_candidates)}")
    print(f"Words     : {words}")
    print(f"Tokens ~  : {tokens}")
