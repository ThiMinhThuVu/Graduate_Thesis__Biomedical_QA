from collections import defaultdict
import os
import json
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from postprocessing import postprocessing

def evaluate_cluster_file(
    input_json_path,
    dict_golden,
    mapping_data,
    golden_mapping
):
    with open(input_json_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    stats = {
        "total_samples": 0,
        "samples_with_evidence": 0,
        "samples_no_evidence": 0,
        "correct_samples": 0,
        "wrong_samples": 0,
    }

    cluster_coverage_list = []
    cluster_count_per_sample = []

    for qidx, clusters in input_data.items():

        if qidx not in dict_golden:
            continue

        golden_list = dict_golden[qidx]["True"]
        if not golden_list:
            continue

        stats["total_samples"] += 1

        total_clusters = 0
        evidence_cluster_count = 0

        has_evidence = False
        is_correct = False

        for label, cluster_sample in clusters.items():

            # if label == "-1":
            #     continue

            total_clusters += 1

            llm_data = cluster_sample.get("llm", {})
            short_answer = llm_data.get("short_answer", "").strip().lower()

            # ✅ Cluster có evidence
            if short_answer and short_answer != "insufficient information":

                evidence_cluster_count += 1
                has_evidence = True

                # ---- normalize ----
                if short_answer in mapping_data:
                    norm_ans = mapping_data[short_answer]
                else:
                    norm_ans = postprocessing(short_answer)
                    if norm_ans in mapping_data:
                        norm_ans = mapping_data[norm_ans]

                for gol in golden_list:
                    if norm_ans.lower() == gol.lower():
                        is_correct = True
                        break

        # ---- Sample-level ----

        cluster_count_per_sample.append(total_clusters)

        coverage = (
            evidence_cluster_count / total_clusters
            if total_clusters > 0 else 0
        )
        cluster_coverage_list.append(coverage)

        if has_evidence:
            stats["samples_with_evidence"] += 1
        else:
            stats["samples_no_evidence"] += 1

        if is_correct:
            stats["correct_samples"] += 1
        else:
            stats["wrong_samples"] += 1

    # ---- accuracy trên câu có evidence ----
    if stats["samples_with_evidence"] > 0:
        accuracy = (
            stats["correct_samples"]
            / stats["samples_with_evidence"]
        )
    else:
        accuracy = 0.0

    avg_cluster_per_sample = (
        sum(cluster_count_per_sample) / len(cluster_count_per_sample)
        if cluster_count_per_sample else 0
    )

    avg_cluster_coverage = (
        sum(cluster_coverage_list) / len(cluster_coverage_list)
        if cluster_coverage_list else 0
    )

    result = {
        **stats,
        "accuracy": accuracy,
        "avg_cluster_per_sample": avg_cluster_per_sample,
        "avg_cluster_coverage": avg_cluster_coverage,
    }

    return result


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
if __name__ == "__main__":
    #ans_file_cluster = r'src\cluster\cache\hdbscan\concat_answer_hdbscan_merge_google_wiki_3.json' #r'src\cluster\cache\kmean\concat_answer_kmean_merge_google_wiki_3.json'
    
    # ans_file_cluster =  r'src\cluster\cache\dynamic\concat_answer_dynamic_merge_google_wiki.json' #r'src\cluster\cache\kmean\only_snippet_answer_kmean_merge_google_wiki_3.json' #r'src\cluster\cache\hdbscan\only_snippet_answer_hdbscan_merge_google_wiki.json'
    ans_file_cluster = r'src\cluster_v2\sai\concat_answer_kmean_merge_google_wiki_3.json'
    metadata = load_json('data/meta_entity_WH.json')
    mapping_data = load_json('data/mapping_answer_gol.json')
    dict_golden = load_json('data/all_answer_golden_56.json')  
    
    golden_mapping = load_json('mapping_golden.json')

    result = evaluate_cluster_file(ans_file_cluster,
                                dict_golden,
                                mapping_data,
                                golden_mapping)
    
    print(f"📌 Tổng số sample                : {result['total_samples']}")
    print(f"📌 Có evidence                   : {result['samples_with_evidence']}")
    print(f"📌 Không có evidence              : {result['samples_no_evidence']}")

    if result['total_samples'] > 0:
        eve_rate = result['samples_with_evidence'] / result['total_samples']
    else:
        eve_rate = 0

    print(f"📌 Tỷ lệ có evidence              : {eve_rate:.4f} ({eve_rate*100:.2f}%)")

    print("-" * 70)

    print(f"✅ Trả lời đúng                  : {result['correct_samples']}")
    print(f"❌ Trả lời sai                   : {result['wrong_samples']}")
    print(f"🎯 Accuracy (trên câu có evidence): {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")

    print("-" * 70)

    print(f"📊 Số cluster trung bình / sample : {result['avg_cluster_per_sample']:.2f}")
    print(f"📊 Độ phủ cluster trung bình      : {result['avg_cluster_coverage']:.4f} ({result['avg_cluster_coverage']*100:.2f}%)")

    print("=" * 70 + "\n")