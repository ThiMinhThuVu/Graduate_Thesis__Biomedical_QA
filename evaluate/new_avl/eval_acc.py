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
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

mapping_data = load_json("data/mapping_answer_gol.json")
dict_context = load_json("all_inf_direct_300.json")

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    dict_golden = json.load(f)

def load_pubmed_model():
    static = StaticEmbedding.from_model2vec(
        "neuml/pubmedbert-base-embeddings-8M"
    )
    return SentenceTransformer(modules=[static])

def detect_source_flags(cluster_sources):
    families = {SOURCE_FAMILY.get(src, src) for src, _ in cluster_sources}
    return "dense" in families, "wiki" in families, "search" in families, families

def cluster_cohesion(embeds):
    centroid = embeds.mean(axis=0, keepdims=True)
    sim = cosine_similarity(embeds, centroid).reshape(-1)
    return float(np.mean(1 - sim))

# =========================
# COLLECT CANDIDATES
# =========================

def collect_wh_candidates(base_paths, dict_golden, run_type=None, topk = 2):
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

            golden_list = qidx_meta[qidx]["golden"]
            if len(golden_list) == 0:
                continue

            # Dense
            if run_type == "DENSE":
                for cand in data.get("create_mcq", []):
                    ans = cand.get("short_answer", "").strip().lower()
                    if ans and ans != "insufficient information":
                        if ans in mapping_data:
                            ans = mapping_data[ans]
                        else:
                            ans = mapping_data.get(postprocessing(ans), postprocessing(ans))
                        candidates.append({
                            "qidx": qidx,
                            "text": ans,
                            "source": "create_MCQ",
                            "run": run_type
                        })
            else:
                source_outputs = (
                    data.get("1", {})
                        .get("0", {})
                        .get("0", {})
                        .get("source_output", {})
                )
                for src, res in source_outputs.items():
                    for i_c, cand in enumerate(res.get("candidate", [])[:topk]):
                        ans = cand.get("short_answer", "").lower()
                        if ans and ans != "insufficient information":
                            if ans in mapping_data:
                                ans = mapping_data[ans]
                            else:
                                ans = mapping_data.get(postprocessing(ans), postprocessing(ans))
                            candidates.append({
                                "qidx": qidx,
                                "text": ans,
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

        list_golden = qidx_meta[qidx]["golden"]
        all_context = dict_context.get(qidx, "")
        check_gol_in_ctx = any(g in all_context for g in list_golden)

        if len(cand_list) == 0:
            fused_results[qidx] = {"abstain": True}
            continue

        if len(cand_list) < 2:
            fused_results[qidx] = {"abstain": True}
            continue
        
        # # ================= SINGLE ANSWER =================
        # if len(cand_list) == 1:

        #     final_ans = cand_list[0]["text"]

        #     if final_ans in mapping_data:
        #         norm_ans = mapping_data[final_ans]
        #     else:
        #         norm_ans = postprocessing(final_ans)
        #         norm_ans = mapping_data.get(norm_ans, norm_ans)

        #     fused_results[qidx] = {
        #         "final_answer": final_ans,
        #         "norm_ans": norm_ans,
        #         "is_correct_norm": norm_ans in list_golden,
        #         "abstain": False,
        #         "check_gol_in_ctx": check_gol_in_ctx
        #     }

            # continue
        

        # ================= MULTI ANSWER CLUSTER =================

        texts = [c["text"] for c in cand_list]
        embeddings = model.encode(texts)

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

            cluster_sources = {
                (cand_list[i]["source"], cand_list[i]["run"])
                for i in indices
            }

            has_dense, has_wiki, has_gg, families = detect_source_flags(cluster_sources)

            raw_clusters.append({
                "indices": indices,
                "evidence_count": evidence_count,
                "has_wiki": has_wiki,
                "has_dense": has_dense,
                "has_gg": has_gg,
                "families": families
            })

            # ================= PRIMARY RULE =================
            # if not (evidence_count >= 3 and has_dense and has_wiki):
            #     continue
            
            if not (evidence_count >= 3 and has_dense and has_wiki):
            # if not (evidence_count >= 3 and has_wiki):
                continue

            cohesion = cluster_cohesion(embeddings[indices])

            if cohesion < best_cohesion:
                best_cluster = indices
                best_cohesion = cohesion

        # ================= FALLBACK RULE =================
        if best_cluster is None:

            if len(raw_clusters) == 2:

                c1, c2 = raw_clusters

                if c1["evidence_count"] != c2["evidence_count"]:

                    larger = c1 if c1["evidence_count"] > c2["evidence_count"] else c2

                    if (larger["has_wiki"] or larger["has_gg"]) and larger["has_dense"]:
                        best_cluster = larger["indices"]
                        best_cohesion = cluster_cohesion(
                            embeddings[best_cluster]
                        )
        # ================= FINAL SINGLE-CLUSTER RULE =================
        if best_cluster is None:

            if len(raw_clusters) == 1:

                only_cluster = raw_clusters[0]

                if (
                    only_cluster["evidence_count"] == 2
                    and only_cluster["has_dense"]
                    and only_cluster["has_wiki"]
                ):
                    best_cluster = only_cluster["indices"]
                    best_cohesion = cluster_cohesion(
                        embeddings[best_cluster]
                    )

        # ================= ABSTAIN =================
        if best_cluster is None:
            fused_results[qidx] = {"abstain": True}
            continue

        # ================= SELECT FINAL ANSWER =================

        centroid = embeddings[best_cluster].mean(axis=0)

        best_idx = min(
            best_cluster,
            key=lambda i: np.linalg.norm(embeddings[i] - centroid)
        )

        final_ans = cand_list[best_idx]["text"]

        if final_ans in mapping_data:
            norm_ans = mapping_data[final_ans]
        else:
            norm_ans = postprocessing(final_ans)
            norm_ans = mapping_data.get(norm_ans, norm_ans)

        fused_results[qidx] = {
            "final_answer": final_ans,
            "norm_ans": norm_ans,
            "is_correct_norm": norm_ans in list_golden,
            "abstain": False,
            "check_gol_in_ctx": check_gol_in_ctx
        }

    return fused_results

# =========================
# FINAL SUMMARY (SELECTED ONLY)
# =========================
def compute_selected_summary(fused_results, qidx_meta):

    total_with_evidence = len(fused_results)  # ✅ đổi sang evidence

    selected = 0
    correct_selected = 0
    total_wrong = 0
    wrong_has_golden_in_ctx = 0
    wrong_no_golden_in_ctx = 0

    for qidx, info in fused_results.items():

        if info.get("abstain", True):
            continue

        selected += 1

        if info.get("is_correct_norm", False):
            correct_selected += 1
        else:
            total_wrong += 1

            if info.get("check_gol_in_ctx", False):
                wrong_has_golden_in_ctx += 1
            else:
                wrong_no_golden_in_ctx += 1

    acc_selected = (
        correct_selected / selected
        if selected else 0.0
    )

    coverage = (
        selected / total_with_evidence
        if total_with_evidence else 0.0
    )

    return {
        "total_with_evidence": total_with_evidence,
        "selected": selected,
        "correct_selected": correct_selected,
        "acc_selected": acc_selected,
        "coverage": coverage,
        "total_wrong": total_wrong,
        "wrong_has_golden_in_ctx": wrong_has_golden_in_ctx,
        "wrong_no_golden_in_ctx": wrong_no_golden_in_ctx
    }


def get_unselected_with_evidence(fused_results, qidx_meta, output_file="zero_shot_wh.json"):

    unselected = {}
    dict_58q = load_json(r'request_batch\batch_output_58_results_o3mcq.json')
    cnt = 0
    for qidx, info in fused_results.items():
        
        # Có evidence vì xuất hiện trong fused_results
        # nhưng không được chọn
        if info.get("abstain", True):
            cnt += 1
            # if qidx in dict_58q:
            #     continue
            question = qidx_meta.get(qidx, {}).get("question", "")

            unselected[qidx] = {
                "question": question
            }
                
    print("Số câu có golden nhưng chưa được chọn: ", cnt)
    print("số câu cần chạy zeroshot: ", len(unselected.keys()))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unselected, f, indent=2, ensure_ascii=False)
    print(output_file)
    
    return unselected

def export_candidates_of_unselected(
    fused_results,
    candidates,
    qidx_meta,
    output_file="unselected_candidates.json"
):
    from collections import defaultdict
    import math

    grouped = defaultdict(list)
    for c in candidates:
        grouped[c["qidx"]].append(c)

    result = {}
    count = 0
    skipped_nan_question = 0

    for qidx, info in fused_results.items():

        if not info.get("abstain", True):
            continue

        question = qidx_meta.get(qidx, {}).get("question", "")

        # ===== BỎ QUESTION NAN =====
        if (
            question is None
            or question == ""
            or (isinstance(question, float) and math.isnan(question))
            or str(question).strip().lower() == "nan"
        ):
            skipped_nan_question += 1
            continue

        golden = qidx_meta.get(qidx, {}).get("golden", [])

        # ===== DÙNG SET ĐỂ LỌC TRÙNG =====
        unique_candidates = {
            (
                cand.get("text", "").strip().lower(),
                cand.get("source", "").strip()
            )
            for cand in grouped.get(qidx, [])
            if cand.get("text")
            and cand.get("text").strip().lower() != "nan"
        }

        # convert lại thành list dict
        clean_candidates = [
            {"text": text, "source": source}
            for text, source in unique_candidates
        ]

        # Nếu sau khi lọc mà không còn candidate hợp lệ → bỏ luôn câu
        if not clean_candidates:
            continue

        result[qidx] = {
            "question": question,
            "golden": golden,
            "candidates": clean_candidates
        }

        count += 1

    print("Số câu abstain hợp lệ:", count)
    print("Số câu bị bỏ do question NaN:", skipped_nan_question)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Đã lưu file:", output_file)

    return result

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
        ["medhop_results_llm_create_mcq"],
        dict_golden,
        "DENSE"
    )
    all_candidates = c1 + c2 + c3
    fused_results = cluster_and_evaluate(all_candidates, meta, model)

    selected_summary = compute_selected_summary(fused_results, meta)
    # unselected_cases = get_unselected_with_evidence(fused_results, meta)
    

    unselected_cases = export_candidates_of_unselected(
        fused_results,
        all_candidates,
        meta,
        output_file="unselected_full_candidates.json"
    )


    print("\n=== SELECTED ANSWER STATISTICS ===")
    print(f"Tổng số câu có evidence   : {selected_summary['total_with_evidence']}")
    print(f"Số câu được chọn          : {selected_summary['selected']}")
    
    print(f"Coverage                  : {selected_summary['coverage']*100:.2f}%")
    print(f"Số câu đúng (trong selected): {selected_summary['correct_selected']}")
    print(f"ACC trên câu được chọn    : {selected_summary['acc_selected']*100:.2f}%")
    print(f"Số câu sai                : {selected_summary['total_wrong']}")
    print(f"   - Có golden trong context    : {selected_summary['wrong_has_golden_in_ctx']}")
    print(f"   - Không có golden trong ctx  : {selected_summary['wrong_no_golden_in_ctx']}")
