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
from postprocessing import postprocessing

# =========================
# CONFIG
# =========================

SOURCE_FAMILY = {
    "google": "search",
    # "wiki": "search",
    "wiki": "wiki",
    "crawl_wiki": "wiki",
    "crawl_ner_wiki": "wiki",
    "crawl_sparql_wikidata": "kg",
    "create_MCQ": "dense"
}

DISTANCE_THRESHOLD = 0.12
GOLDEN_PATH = "data/all_answer_golden_61.json"
CSV_OUT = "evaluate/acc/fusion_trace.csv"


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

mapping_data = load_json("wiki_heading_mapping.json")

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    dict_golden = json.load(f)

# =========================
# MODEL
# =========================

def load_pubmed_model():
    static = StaticEmbedding.from_model2vec(
        "neuml/pubmedbert-base-embeddings-8M"
    )
    return SentenceTransformer(modules=[static])

# =========================
# DATA COLLECTION
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
            if qidx not in dict_golden:
                continue

            if data["type_q"] != "direct" or data["target_answer_type"] == "correctness":
                continue

            if qidx not in qidx_meta:
                qidx_meta[qidx] = {
                    "golden": [g.lower().strip() for g in dict_golden[qidx]["True"]]
                }

            # Dense
            if run_type == "DENSE":
                for cand in data.get("create_mcq", []):
                    ans = cand.get("short_answer", "").strip().lower()
                    if ans and ans != "insufficient information":
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
                            candidates.append({
                                "qidx": qidx,
                                "text": ans,
                                "source": src,
                                "run": f"{run_type}_{i_c}"
                            })

    return candidates, qidx_meta

# =========================
# RULE UTILS
# =========================

def detect_source_flags(cluster_sources):
    families = {SOURCE_FAMILY.get(src, src) for src, _ in cluster_sources}
    # return "dense" in families, "wiki" in families, families
    return "dense" in families, "wiki" in families, families
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

        if len(cand_list) < 2:
            fused_results[qidx] = {
                "abstain": True,
                "cluster_sources": [(c["source"], c["run"]) for c in cand_list],
                "evidence_count": len(cand_list),
                "num_source_families": 1
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

        best_label = np.argmax(np.bincount(labels))
        best_indices = [i for i, l in enumerate(labels) if l == best_label]
        evidence_count = len(best_indices)

        cluster_sources = {(cand_list[i]["source"], cand_list[i]["run"]) for i in best_indices}
        has_dense, has_wiki, families = detect_source_flags(cluster_sources)
        num_families = len(families)
        # ===== RULE =====
        # if not ((evidence_count >= 4)):
        # if not ((evidence_count == 3 and num_families >= 3)):
        # if not ((evidence_count >= 2 and num_families >= 2)):
        # if not ((evidence_count >= 3 and num_families >= 2)): 
        # if not ((evidence_count >= 3) or (evidence_count >= 2 and has_dense)):
        if not ((evidence_count >= 4) or (evidence_count >= 3 and num_families >= 3) or (evidence_count >= 2 and has_dense and has_wiki)):
        # if not ((evidence_count >= 4) or (evidence_count >= 3 and has_dense and has_wiki)):
            fused_results[qidx] = {
                "abstain": True,
                "cluster_sources": list(cluster_sources),
                "evidence_count": evidence_count,
                "num_source_families": num_families
            }
            continue
        
        # ===== SELECT =====
        centroid = embeddings[best_indices].mean(axis=0)
        best_idx = min(best_indices, key=lambda i: np.linalg.norm(embeddings[i] - centroid))
        final_ans = cand_list[best_idx]["text"]

        norm_ans = postprocessing(final_ans)
        # if norm_ans in mapping_data:
        #     norm_ans = mapping_data[norm_ans]

        if qidx not in qidx_meta:
            continue
        
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
            "cluster_sources": list(cluster_sources),
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
            "final_run", "cluster_sources"
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
                "; ".join(f"{s}|{r}" for s, r in info["cluster_sources"])
            ])

    print("Saved:", path)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    model = load_pubmed_model()

    c1, meta = collect_wh_candidates(["medhop_results_mcq_2_TFIDF1000"], dict_golden, "TFIDF1000")
    
    # c2, meta = collect_wh_candidates(["medhop_results_mcq_3_TFIDF300"], dict_golden, "TFIDF300")
    # c2, meta = collect_wh_candidates(["medhop_results_no_mcq_TFIDF300"], dict_golden, "TFIDF300")
    c3, _ = collect_wh_candidates(["medhop_results_no_mcq_TFIDF300"], dict_golden, "DENSE")

    # stats = cluster_and_evaluate(c1 + c2 + c3, meta, model)
    stats = cluster_and_evaluate(c1 + c3, meta, model)

    # print("\n=== FINAL RESULT ===")
    # print(f"Accuracy (raw)  : {stats['accuracy']*100:.2f}%")
    # print(f"Accuracy (norm) : {stats['norm_accuracy']*100:.2f}%")
    # print(f"Coverage        : {stats['coverage']*100:.2f}%")
    # print(f"Answered        : {stats['answered']} / {stats['total_questions']}")

    export_fusion_csv(stats["details"], CSV_OUT)

    print("\n=== FINAL RESULT ===")

    raw_accuracy = stats["accuracy"]
    norm_accuracy = stats["norm_accuracy"]
    coverage = stats["coverage"]

    answered = stats["answered"]
    total_questions = stats["total_questions"]

    qualified = stats["evidence_selected"]      # pass rule
    correct_raw = stats["correct_selected"]
    correct_norm = stats["norm_correct"]

    print(f"Coverage         : {coverage*100:.2f}%")
    print(f"Answered         : {answered} / {total_questions}")

    print("\nEvidence-qualified questions:")
    print(
        f"  Correct (raw)         : {correct_raw} / {answered}\n"
        f"  Correct (norm)        : {correct_norm} / {answered}\n"
        f"  Rate (raw)            : {raw_accuracy*100:.2f}%\n"
        f"  Rate (norm)           : {norm_accuracy*100:.2f}%"
    )

    print("\nAccuracy by evidence:")
    for ev in sorted(stats["acc_by_evidence"]):
        d = stats["acc_by_evidence"][ev]
        acc = d["correct"] / d["total"] if d["total"] else 0
        print(f"  Evidence={ev:2d} | Total={d['total']:3d} | Acc={acc*100:.2f}%")
