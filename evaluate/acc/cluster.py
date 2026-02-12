import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.models import StaticEmbedding
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

# 1. Cấu hình hằng số
SOURCE_ORDER = ["google", "wiki", "crawl_wiki", "crawl_ner_wiki", "crawl_sparql_wikidata"]

SOURCE_FAMILY = {
    "google": "search",
    "wiki": "wiki",
    "crawl_wiki": "wiki",
    "crawl_ner_wiki": "wiki",
    "crawl_sparql_wikidata": "kg",
    "create_MCQ": "dense"
}


with open('data/all_answer_golden_61.json', "r", encoding="utf-8") as f:
    dict_golden = json.load(f)
    
def load_pubmed_model():
    """Khởi tạo model PubMedBERT qua model2vec (Tốc độ & Chuyên biệt Y sinh)"""
    static = StaticEmbedding.from_model2vec("neuml/pubmedbert-base-embeddings-8M")
    return SentenceTransformer(modules=[static])
def collect_wh_candidates(base_paths, dict_golden, run_type=None):
    """
    Thu thập candidate answers cho WH questions
    KHÔNG clustering
    """
    candidates = []
    qidx_meta = {}

    for path in base_paths:
        all_ans_path = os.path.join(path, "all_ans")
        if not os.path.exists(all_ans_path):
            continue

        for fname in os.listdir(all_ans_path):
            if not fname.endswith(".json"):
                continue

            with open(os.path.join(all_ans_path, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)

            qidx = data.get("QIDX")
            if qidx not in dict_golden:
                continue

            if data['type_q'] != 'direct' or data['target_answer_type'] == 'correctness':
                continue

            # lưu golden & meta (chỉ 1 lần / qidx)
            if qidx not in qidx_meta:
                qidx_meta[qidx] = {
                    "golden": [g.lower().strip() for g in dict_golden[qidx]['True']]
                }

            if run_type == 'DENSE':
                create_mcq = data['create_mcq']
                for cand in create_mcq:
                    short_ans = cand['short_ans']
                    if not short_ans or short_ans == "insufficient information":
                        continue

                    candidates.append({
                        "qidx": qidx,
                        "text": short_ans,
                        "source": 'create_MCQ',
                        "run": run_type
                    })
            else:
                source_outputs = (
                    data.get('1', {})
                        .get('0', {})
                        .get('0', {})
                        .get('source_output', {})
                )

                for src_name, res in source_outputs.items():
                    short_ans = res.get("short_answer", "").strip().lower()

                    if not short_ans or short_ans == "insufficient information":
                        continue

                    candidates.append({
                        "qidx": qidx,
                        "text": short_ans,
                        "source": src_name,
                        "run": run_type
                    })

    return candidates, qidx_meta
def cluster_and_evaluate(candidates, qidx_meta, model, distance_threshold=0.12):
    grouped = defaultdict(list)
    for c in candidates:
        grouped[c["qidx"]].append(c)

    total_questions = 0
    total_correct = 0

    fused_results = {}
    acc_by_evidence = defaultdict(lambda: {"correct": 0, "total": 0})

    for qidx, cand_list in grouped.items():
        total_questions += 1

        # =========================
        # CASE 1: CHỈ CÓ 1 CANDIDATE
        # =========================
        if len(cand_list) == 1:
            final_candidate = cand_list[0]
            final_ans = final_candidate["text"]
            evidence_count = 1

            is_correct = any(final_ans == g for g in qidx_meta[qidx]["golden"])
            if is_correct:
                total_correct += 1

            acc_by_evidence[1]["total"] += 1
            if is_correct:
                acc_by_evidence[1]["correct"] += 1

            fused_results[qidx] = {
                "final_answer": final_ans,
                "is_correct": is_correct,
                "evidence_count": 1,
                "final_source": final_candidate["source"],
                "final_run": final_candidate["run"],
                "cluster_sources": [(final_candidate["source"], final_candidate["run"])]
            }
            continue  # ⚠️ RẤT QUAN TRỌNG

        # =========================
        # CASE 2: ≥ 2 CANDIDATES → CLUSTER
        # =========================
        texts = [c["text"] for c in cand_list]
        embeddings = model.encode(texts)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average"
        ).fit(embeddings)

        labels = clustering.labels_
        counts = np.bincount(labels)
        best_label = np.argmax(counts)

        best_indices = [i for i, l in enumerate(labels) if l == best_label]
        evidence_count = len(best_indices)

        #THUVTM
        
        # # =========================
        # # HARD RULE: consensus + source diversity
        # # =========================
        # cluster_sources = {(cand_list[i]["source"], cand_list[i]["run"]) for i in best_indices}

        # source_families = {
        #     SOURCE_FAMILY.get(src, src)
        #     for src, _ in cluster_sources
        # }
        # num_families = len(source_families)

        # # RULE CỨNG
        # if evidence_count < 3 or num_families < 2:
        #     # ABSTAIN
        #     acc_by_evidence[evidence_count]["total"] += 1

        #     fused_results[qidx] = {
        #         "final_answer": None,
        #         "is_correct": False,
        #         "evidence_count": evidence_count,
        #         "final_source": None,
        #         "final_run": None,
        #         "cluster_sources": list(cluster_sources),
        #         "abstain": True,
        #         "num_source_families": num_families
        #     }
        #     continue


        #THUVTM
        centroid = np.mean(embeddings[best_indices], axis=0)
        best_local_idx = min(
            range(len(best_indices)),
            key=lambda i: np.linalg.norm(
                embeddings[best_indices[i]] - centroid
            )
        )
        best_idx = best_indices[best_local_idx]

        final_candidate = cand_list[best_idx]
        final_ans = final_candidate["text"]

        is_correct = any(final_ans == g for g in qidx_meta[qidx]["golden"])
        if is_correct:
            total_correct += 1

        acc_by_evidence[evidence_count]["total"] += 1
        if is_correct:
            acc_by_evidence[evidence_count]["correct"] += 1

        fused_results[qidx] = {
            "final_answer": final_ans,
            "is_correct": is_correct,
            "evidence_count": evidence_count,
            "final_source": final_candidate["source"],
            "final_run": final_candidate["run"],
            "cluster_sources": list(
                {(cand_list[i]["source"], cand_list[i]["run"]) for i in best_indices}
            )
        }

    accuracy = total_correct / total_questions if total_questions > 0 else 0.0

    return {
        "total_questions": total_questions,
        "correct_questions": total_correct,
        "accuracy": accuracy,
        "acc_by_evidence": acc_by_evidence,
        "details": fused_results
    }

import csv

def export_fusion_csv(fused_results, csv_path="evaluate/acc/fusion_trace.csv"):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "QIDX",
            "final_answer",
            "is_correct",
            "evidence_count",
            "final_source",
            "final_run",
            "cluster_sources"
        ])

        for qidx, info in fused_results.items():
            writer.writerow([
                qidx,
                info["final_answer"],
                info["is_correct"],
                info["evidence_count"],
                info["final_source"],
                info["final_run"],
                "; ".join([f"{s}|{r}" for s, r in info["cluster_sources"]])
            ])
    print("Save: ", csv_path)

if __name__ == "__main__":
    model = load_pubmed_model()

    all_candidates = []

    c1, meta = collect_wh_candidates(
        ["medhop_results_mcq_2_TFIDF1000"],
        dict_golden,
        run_type="TFIDF1000"
    )

    c2, _ = collect_wh_candidates(
        ["medhop_results_mcq_3_TFIDF300"],
        dict_golden,
        run_type="TFIDF300"
    )

    c3, _ = collect_wh_candidates(
        ["medhop_results_mcq_dense"],
        dict_golden,
        run_type="DENSE"
    )

    all_candidates.extend(c1)
    all_candidates.extend(c2)
    all_candidates.extend(c3)

    stats = cluster_and_evaluate(all_candidates, meta, model)

    print("\nAccuracy theo evidence_count:")
    for ev in sorted(stats["acc_by_evidence"]):
        d = stats["acc_by_evidence"][ev]
        acc = d["correct"] / d["total"]
        print(f"  Evidence = {ev:2d} | "
            f"Total = {d['total']:3d} | "
            f"Correct = {d['correct']:3d} | "
            f"Acc = {acc*100:.2f}%")
    export_fusion_csv(stats["details"])
