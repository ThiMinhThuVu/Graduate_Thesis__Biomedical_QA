
from collections import defaultdict
import os
import json
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from postprocessing import postprocessing


SOURCE_ORDER = [
    "google",
    "wiki",
    "crawl_wiki",
    "crawl_ner_wiki",
    "crawl_sparql_wikidata"
]

with open('data/meta_entity_WH.json', "r", encoding="utf-8") as f:
# with open('data/meta_entity_YN.json', "r", encoding="utf-8") as f:
    metadata = json.load(f)
        
# with open('data/mapping_answer_gol_old1.json', "r", encoding="utf-8") as f:
#     mapping_data = json.load(f)
with open('data/mapping_answer_gol.json', "r", encoding="utf-8") as f:
    mapping_data = json.load(f)
    
with open('data/all_answer_golden_56.json', "r", encoding="utf-8") as f:
    dict_golden = json.load(f)

def normalize_source(src):
    return src.strip().lower() if src else "unknown"


def collect_wh_candidates(base_paths, dict_golden, run_type=None, topk = 3):
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

def analyze_any_source_correct(candidates, qidx_meta):
    """
    candidates: output từ collect_wh_candidates
    qidx_meta: metadata chứa golden answer
    """

    qidx_data = defaultdict(list)

    # Gom theo QIDX
    for c in candidates:
        qidx_data[c["qidx"]].append(c)

    total_samples = 0
    samples_with_evidence = 0
    correct_samples = 0

    for qidx, cand_list in qidx_data.items():

        if qidx not in qidx_meta:
            continue

        golden_list = qidx_meta[qidx]["golden"]
        if not golden_list:
            continue

        total_samples += 1

        has_evidence = False
        sample_correct = False

        for cand in cand_list:

            ans = cand["text"]

            if not ans:
                continue

            has_evidence = True

            if any(ans == g for g in golden_list):
                sample_correct = True
                break

        if has_evidence:
            samples_with_evidence += 1

        if sample_correct:
            correct_samples += 1

    accuracy_any_source = (
        correct_samples / samples_with_evidence
        if samples_with_evidence > 0 else 0.0
    )

    return {
        "total_samples": total_samples,
        "samples_with_evidence": samples_with_evidence,
        "correct_samples": correct_samples,
        "accuracy_any_source": accuracy_any_source
    }

if __name__ == "__main__":

    c1, meta = collect_wh_candidates(
            ["medhop_results_mcq_2_TFIDF1000"],
            dict_golden,
            "TFIDF1000"
        )

    c2, _ = collect_wh_candidates(
        ["medhop_results_mcq_3_TFIDF300"],
        dict_golden,
        "TFIDF300",
        topk=1
    )

    c3, _ = collect_wh_candidates(
        ["medhop_results_llm_create_mcq"],
        dict_golden,
        "DENSE"
    )
    # all_candidates = c1 + c2 + c3
    
    all_candidates = c2
    any_src_stats = analyze_any_source_correct(all_candidates, meta)
    
    
    print("\n=== ANY SOURCE CORRECT (SAMPLE LEVEL) ===")
    print(f"Total samples            : {any_src_stats['total_samples']}")
    print(f"Samples with evidence    : {any_src_stats['samples_with_evidence']}")
    print(f"Correct samples          : {any_src_stats['correct_samples']}")
    print(f"Accuracy (any source)    : {any_src_stats['accuracy_any_source']:.2%}")
        