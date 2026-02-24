import json
import os
import pandas as pd
from postprocessing import postprocessing
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    answer = 6
    BASE_DIR = f"extract_ner_kmean/answer{answer}"
    # direct_result_WH = load_json(f"{BASE_DIR}/CT6_final_answer_no_mapping_direct_WH_WMASS0.7_WCONC0.3.json")
    # direct_result_YN = load_json(f"{BASE_DIR}/CT6_final_answer_direct_YN_WMASS0.7_WCONC0.3.json")
    
    direct_result_WH = load_json(fr'src\cluster\final\WH_answer_CT6_#5.json')
    direct_result_YN = load_json(fr'src\cluster\final\YN_answer_CT6_#5.json')
    
    zero_shot_direct = load_json("data/cache/O3_zeroshot_ans_cache_direct.json")
    # zero_shot_direct = load_json("O3_zeroshot_ans_cache.json")
    # zero_shot_seq = load_json("data/cache/O3_zeroshot_ans_cache_sequence.json")
    golden_data = load_json("data/all_answer_golden_56.json")
    metadata = load_json("data/metadata_medhop.json") 
    mapping_data = load_json("data/mapping_answer_gol.json")
 
    # list_id_miss = load_json("statistic/CT2B/CT2B_WH_id_miss_direct.json")
    # zero_shot_true = 0
    # for qidx in list_id_miss:
    #     zero_shot = zero_shot_direct[qidx]["short_answer"]
    #     if zero_shot in mapping_data:
    #         final_answer = mapping_data[zero_shot]
    #     else:
    #         final_answer = postprocessing(zero_shot).lower()
    #         if final_answer in mapping_data:
    #             final_answer = mapping_data[final_answer]

    #     qidx = str(qidx)
    #     golden = golden_data[qidx]["True"]

    #     if len(golden) == 0:
    #         continue

    #     if final_answer.lower() in golden:
    #         zero_shot_true += 1
    # print(zero_shot_true)

    target_ans_type_dict = {
        "correctness": {"total": 0, "acc": 0},
        "other": {"total": 0, "acc": 0},
        "chromosome": {"total": 0, "acc": 0},
        "condition": {"total": 0, "acc": 0},
        "gene": {"total": 0, "acc": 0},
        "disease": {"total": 0, "acc": 0},
        "medication": {"total": 0, "acc": 0},
        "disorder": {"total": 0, "acc": 0},
        "protein": {"total": 0, "acc": 0},
        "syndrome": {"total": 0, "acc": 0},
        "person": {"total": 0, "acc": 0},
        "gene mutation": {"total": 0, "acc": 0},
        "year": {"total": 0, "acc": 0},
        "symptom": {"total": 0, "acc": 0},
        "virus": {"total": 0, "acc": 0},
        "anatomical structure": {"total": 0, "acc": 0},
        "drug": {"total": 0, "acc": 0},
        "muscle": {"total": 0, "acc": 0},
        "number of amino acids": {"total": 0, "acc": 0},
        "choice": {"total": 0, "acc": 0},
        "enzyme": {"total": 0, "acc": 0},
        "tumor": {"total": 0, "acc": 0},
        "nerve": {"total": 0, "acc": 0},
        "antibiotic": {"total": 0, "acc": 0},
        "surgical procedure": {"total": 0, "acc": 0},
        "organ": {"total": 0, "acc": 0},
        "cancer": {"total": 0, "acc": 0}
    }

    for qidx in direct_result_WH:
        if "final_answer" in direct_result_WH[qidx]:
            final_answer = direct_result_WH[qidx]["final_answer"].lower()
        else:
            zero_shot = zero_shot_direct[qidx]["1"]["0"]["0"]["short_answer"].lower()
            # if zero_shot in mapping_data:
            #     final_answer = mapping_data[zero_shot]
            # else:
            #     final_answer = postprocessing(zero_shot)
            #     if final_answer in mapping_data:
            #         final_answer = mapping_data[final_answer]
            final_answer = postprocessing(zero_shot).lower()
        
        target_ans_type = metadata[qidx]["target_answer_type"]
        if target_ans_type not in target_ans_type_dict:
            target_ans_type = "other"

        golden = golden_data[qidx]["True"]

        if len(golden) == 0:
            continue

        target_ans_type_dict[target_ans_type]["total"] += 1
        if final_answer in golden:
            target_ans_type_dict[target_ans_type]["acc"] += 1

    for qidx in direct_result_YN:
        if "final_answer" in direct_result_YN[qidx]:
            final_answer = direct_result_YN[qidx]["final_answer"].lower()
        else:
            zero_shot = zero_shot_direct[qidx]["1"]["0"]["0"]["short_answer"].lower()
            # if zero_shot in mapping_data:
            #     final_answer = mapping_data[zero_shot]
            # else:
            #     final_answer = postprocessing(zero_shot)
            #     if final_answer in mapping_data:
            #         final_answer = mapping_data[final_answer]
            final_answer = postprocessing(zero_shot).lower()

        target_ans_type = metadata[qidx]["target_answer_type"]
        if target_ans_type not in target_ans_type_dict:
            target_ans_type = "other"

        golden = golden_data[qidx]["True"]

        if len(golden) == 0:
            continue

        target_ans_type_dict[target_ans_type]["total"] += 1
        if final_answer in golden:
            target_ans_type_dict[target_ans_type]["acc"] += 1

    all_type_total = sum([target_ans_type_dict[target_ans_type]["acc"] for target_ans_type in target_ans_type_dict])
    # print(all_type_total - target_ans_type_dict["correctness"]["acc"])

    df = pd.DataFrame.from_dict(target_ans_type_dict, orient="index").reset_index()
    df = df.rename(columns={"index": "target_ans_type"})
    df.to_csv(f"{BASE_DIR}/ct6_target_type_no_mapping.csv", index=False) 