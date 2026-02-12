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
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

mapping_data = load_json("data/mapping_answer_gol.json")

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    dict_golden = json.load(f)

def load_pubmed_model():
    static = StaticEmbedding.from_model2vec(
        "neuml/pubmedbert-base-embeddings-8M"
    )
    return SentenceTransformer(modules=[static])

def detect_source_flags(cluster_sources):
    families = {SOURCE_FAMILY.get(src, src) for src, _ in cluster_sources}
    return "dense" in families, "wiki" in families, families

def cluster_cohesion(embeds):
    """
    Average cosine distance to centroid
    """
    centroid = embeds.mean(axis=0, keepdims=True)
    sim = cosine_similarity(embeds, centroid).reshape(-1)
    return float(np.mean(1 - sim))

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
                
            if len(golden_list) == 0:
                continue

            # Dense
            if run_type == "DENSE":
                for cand in data.get("create_mcq", []):
                    ans = cand.get("short_answer", "").strip().lower()
                    if ans and ans != "insufficient information":
                        # ans = postprocessing(ans)
                        candidates.append({
                            "qidx": qidx,
                            "text": ans,
                            "source": "create_MCQ",
                            "run": run_type
                        })

            # Sparse
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
                        if ans and ans != "insufficient information":
                            # ans = postprocessing(ans)
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

    total_questions = len(grouped)
    answered = correct = 0
    norm_correct = norm_total = 0
    acc_by_evidence = defaultdict(lambda: {"correct": 0, "total": 0})
    fused_results = {}

    for qidx, cand_list in grouped.items():

        # if len(cand_list) < 2:
        #     fused_results[qidx] = {
        #         "abstain": True,
        #         "cluster_sources": [(c["source"], c["run"]) for c in cand_list],
        #         "evidence_count": len(cand_list),
        #         "num_source_families": 1
        #     }
        #     continue
        
        if len(cand_list) == 1:
            c = cand_list[0]
            family = SOURCE_FAMILY.get(c["source"], c["source"])

            # if family in ["dense"]:
            final_ans = c["text"]
            
            #THUVTM
            final_ans = postprocessing(final_ans)
            norm_ans = postprocessing(final_ans)

            is_correct = final_ans in qidx_meta[qidx]["golden"]
            is_correct_norm = norm_ans in qidx_meta[qidx]["golden"]

            answered += 1
            acc_by_evidence[1]["total"] += 1
            if is_correct:
                correct += 1
                acc_by_evidence[1]["correct"] += 1

            norm_total += 1
            if is_correct_norm:
                norm_correct += 1

            fused_results[qidx] = {
                "final_answer": final_ans,
                "norm_ans": norm_ans,
                "is_correct": is_correct,
                "is_correct_norm": is_correct_norm,
                "final_source": c["source"],
                "final_run": c["run"],
                "evidence_count": 1,
                "num_source_families": 1,
                "cluster_sources": [(c["source"], c["run"])],
                "cohesion": 0.0,
                "abstain": False
            }
            # else:
            #     fused_results[qidx] = {
            #         "abstain": True,
            #         "cluster_sources": [(c["source"], c["run"])],
            #         "evidence_count": 1,
            #         "num_source_families": 1
            #     }
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
            evidence_count = len(indices)

            cluster_sources = {(cand_list[i]["source"], cand_list[i]["run"]) for i in indices}
            has_dense, has_wiki, families = detect_source_flags(cluster_sources)
            num_families = len(families)

            # if not ((evidence_count >= 2)): # or (evidence_count >= 3 and has_dense and has_wiki)):
            if not ((evidence_count >= 6) or (evidence_count >= 4 and has_dense and has_wiki)):
            # if not ((evidence_count >= 4) or (evidence_count >= 3 and has_dense and has_wiki) or (evidence_count == 2 and has_wiki)):
            # if not (
            #     (evidence_count >= 4) or
            #     (evidence_count >= 3 and (num_families >= 3 or has_wiki)) or
            #     (evidence_count >= 2 and has_dense and has_wiki) #and has_wiki
            # ):
                continue

            cohesion = cluster_cohesion(embeddings[indices])

            cluster_info.append({
                "indices": indices,
                "cohesion": cohesion,
                "evidence_count": evidence_count,
                "families": families,
                "cluster_sources": cluster_sources
            })

        if not cluster_info:
            fused_results[qidx] = {
                "abstain": True,
                "cluster_sources": [],
                "evidence_count": 0,
                "num_source_families": 0
            }
            continue

        best_cluster = min(cluster_info, key=lambda x: x["cohesion"])
        best_indices = best_cluster["indices"]
        evidence_count = best_cluster["evidence_count"]
        families = best_cluster["families"]

        centroid = embeddings[best_indices].mean(axis=0)
        best_idx = min(best_indices, key=lambda i: np.linalg.norm(embeddings[i] - centroid))

        final_ans = cand_list[best_idx]["text"]
        norm_ans = postprocessing(final_ans)

        is_correct = final_ans in qidx_meta[qidx]["golden"]
        is_correct_norm = norm_ans in qidx_meta[qidx]["golden"]

        answered += 1
        acc_by_evidence[evidence_count]["total"] += 1
        if is_correct:
            correct += 1
            acc_by_evidence[evidence_count]["correct"] += 1

        norm_total += 1
        if is_correct_norm:
            norm_correct += 1

        fused_results[qidx] = {
            "final_answer": final_ans,
            "norm_ans": norm_ans,
            "is_correct": is_correct,
            "is_correct_norm": is_correct_norm,
            "final_source": cand_list[best_idx]["source"],
            "final_run": cand_list[best_idx]["run"],
            "evidence_count": evidence_count,
            "num_source_families": len(families),
            "cluster_sources": list(best_cluster["cluster_sources"]),
            "cohesion": best_cluster["cohesion"],
            "abstain": False
        }

    return {
        "accuracy": correct / answered if answered else 0.0,
        "norm_accuracy": norm_correct / norm_total if norm_total else 0.0,
        "coverage": answered / total_questions if total_questions else 0.0,
        "answered": answered,
        "total_questions": total_questions,
        "evidence_selected": answered,
        "correct_selected": correct,
        "norm_correct": norm_correct,
        "acc_by_evidence": acc_by_evidence,
        "details": fused_results
    }

# =========================
# EXPORT CSV
# =========================

def export_fusion_csv(fused_results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "QIDX", "final_answer", "norm_ans",
            "is_correct", "is_correct_norm",
            "abstain", "evidence_count",
            "num_source_families", "final_source",
            "final_run", "cohesion", "cluster_sources"
        ])

        for qidx, info in fused_results.items():
            writer.writerow([
                qidx,
                info.get("final_answer", ""),
                info.get("norm_ans", ""),
                info.get("is_correct", ""),
                info.get("is_correct_norm", ""),
                info["abstain"],
                info["evidence_count"],
                info["num_source_families"],
                info.get("final_source", ""),
                info.get("final_run", ""),
                info.get("cohesion", ""),
                "; ".join(f"{s}|{r}" for s, r in info["cluster_sources"])
            ])

    print("Saved:", path)
def export_empty_golden_csv(qidx_meta, fused_results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "QIDX",
            "Question",
            "Short Answer",
            "Long Answer"
        ])

        for qidx, meta in qidx_meta.items():
            if meta["has_golden"]:
                continue
            golden_false = dict_golden[qidx]['False']
            # lấy final normalized answer nếu có
            short_ans = ""
            if qidx in fused_results and not fused_results[qidx].get("abstain", False):
                short_ans = fused_results[qidx].get("norm_ans", "")

            if short_ans != "" and short_ans not in golden_false:
                writer.writerow([
                    qidx,
                    meta.get("question", ""),
                    short_ans,
                    ""   # Long Answer để rỗng
                ])

    print("Saved empty-golden CSV:", out_path)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    EMPTY_GOLDEN_CSV = "evaluate/acc/empty_golden_questions.csv"

    model = load_pubmed_model()

    c1, meta = collect_wh_candidates(
        ["medhop_results_mcq_2_TFIDF1000"],
        dict_golden,
        "TFIDF1000"
    )
    c2, _ = collect_wh_candidates(["medhop_results_mcq_3_TFIDF300"], dict_golden, "TFIDF300")
    
    c3, _ = collect_wh_candidates(
        ["medhop_results_no_mcq_TFIDF300"],
        dict_golden,
        "DENSE"
    )

    stats = cluster_and_evaluate(c1 + c2 + c3, meta, model)
    correct_raw = stats["correct_selected"]
    correct_norm = stats["norm_correct"]

    export_fusion_csv(stats["details"], CSV_OUT)
    export_empty_golden_csv(meta, stats["details"], EMPTY_GOLDEN_CSV)
    
    print("\n=== FINAL RESULT ===")
    print(f"Coverage        : {stats['coverage']*100:.2f}%")
    print(f"Answered        : {stats['answered']} / {stats['total_questions']}")
    
    print(
        f"  Correct (raw)         : {correct_raw} \n"
        f"  Correct (norm)        : {correct_norm} \n")
    print(f"Accuracy (raw)  : {stats['accuracy']*100:.2f}%")
    print(f"Accuracy (norm) : {stats['norm_accuracy']*100:.2f}%")
    
    print("\nAccuracy by evidence:")
    for ev in sorted(stats["acc_by_evidence"]):
        d = stats["acc_by_evidence"][ev]
        acc = d["correct"] / d["total"] if d["total"] else 0
        print(f"  Evidence={ev:2d} | Total={d['total']:3d} | Acc={acc*100:.2f}%")
