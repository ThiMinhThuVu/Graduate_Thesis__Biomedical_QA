import json
import os
import pandas as pd

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    seq_result = load_json("statistic/CT2B/CT2B_final_answer_seq.json")
    direct_result_WH = load_json("statistic\CT2B\CT2B_final_answer_direct_WH.json")
    direct_result_YN = load_json("statistic\CT2B\CT2B_final_answer_direct_YN.json")
    golden_data = load_json("data/all_answer_golden_56.json")
    metadata = load_json("data/metadata_medhop.json")

    target_ans_type_dict = {}
    target_ans_type_key = [
        "correctness",
        "chromosome", 
        "condition",
        "gene",
        "disease",
        "medication",
        "disorder",
        "protein",
        "syndrome",
        "person",
        "gene mutation",
        "year",
        "symptom",
        "virus",
        "anatomical structure",
        "drug",
        "muscle",
        "number of amino acids",
        "choice",
        "enzyme",
        "tumor",
        "nerve",
        "antibiotic",
        "surgical procedure",
        "organ",
        "cancer"
    ]

    for qidx in seq_result:
        final_answer = seq_result[qidx]["final_answer"].lower()
        target_ans_type = seq_result[qidx]["target_ans_type_last_subq"].lower()
        if target_ans_type not in target_ans_type_key:
            target_ans_type = "other"

        if target_ans_type not in target_ans_type_dict:
            target_ans_type_dict[target_ans_type] = {
                "direct_total": 0,
                "direct_true": 0,
                "seq_total": 0,
                "seq_true": 0
            }

        golden = golden_data[qidx]["True"]
        target_ans_type_dict[target_ans_type]["seq_total"] += 1
        if final_answer in golden:
            target_ans_type_dict[target_ans_type]["seq_true"] += 1

    for qidx in direct_result_WH:
        final_answer = direct_result_WH[qidx].get("final_answer", " ")
        target_ans_type = metadata[qidx]["target_answer_type"]
        if target_ans_type not in target_ans_type_key:
            target_ans_type = "other"

        if target_ans_type not in target_ans_type_dict:
            target_ans_type_dict[target_ans_type] = {
                "direct_total": 0,
                "direct_true": 0,
                "seq_total": 0,
                "seq_true": 0
            }

        golden = golden_data[qidx]["True"]
        target_ans_type_dict[target_ans_type]["direct_total"] += 1
        if final_answer in golden:
            target_ans_type_dict[target_ans_type]["direct_true"] += 1

    for qidx in direct_result_YN:
        final_answer = direct_result_YN[qidx].get("final_answer", " ")
        target_ans_type = metadata[qidx]["target_answer_type"]
        if target_ans_type not in target_ans_type_key:
            target_ans_type = "other"

        if target_ans_type not in target_ans_type_dict:
            target_ans_type_dict[target_ans_type] = {
                "direct_total": 0,
                "direct_true": 0,
                "seq_total": 0,
                "seq_true": 0
            }

        golden = golden_data[qidx]["True"]
        target_ans_type_dict[target_ans_type]["direct_total"] += 1
        if final_answer in golden:
            target_ans_type_dict[target_ans_type]["direct_true"] += 1

    df = pd.DataFrame.from_dict(target_ans_type_dict, orient="index").reset_index()
    df = df.rename(columns={"index": "target_ans_type"})
    df.to_csv("statistic/stat_target_ans_type.csv", index=False)