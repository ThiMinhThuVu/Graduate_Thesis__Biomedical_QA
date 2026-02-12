import os
import json
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from postprocessing import postprocessing

NO_EVIDENCE = "insufficient information"

# ======================
# UTILS
# ======================
def normalize(text: str) -> str:
    if not text:
        return ""
    return text.strip().lower()

# ======================
# CORE TASK
# ======================
def build_answer_dict(json_folder, dict_golden, mapping_data):
    """
    Return:
    {
      qidx: {
        "golden": [...],
        "cand": {
            norm_answer: {
                "is_correct": bool,
                "post_ans": str,
                "post_ans_correct": bool
            }
        }
      }
    }
    """
    result = {}

    for fname in os.listdir(json_folder):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(json_folder, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        qidx = str(data.get("QIDX"))
        if data.get("target_answer_type") == "correctness":
            continue
        if qidx not in dict_golden:
            continue

        golden_list = [
            normalize(g) for g in dict_golden[qidx].get("True", []) if g
        ]
        if not golden_list:
            continue

        source_outputs = (
            data.get("1", {})
                .get("0", {})
                .get("0", {})
                .get("source_output", {})
        )

        cand_dict = {}

        for _, src_data in source_outputs.items():
            raw_ans = normalize(src_data.get("short_answer", ""))

            if not raw_ans or raw_ans == NO_EVIDENCE:
                continue

            # alias mapping
            norm_ans = mapping_data.get(raw_ans, raw_ans)

            is_correct = norm_ans in golden_list

            post_ans = normalize(postprocessing(norm_ans))
            post_ans = mapping_data.get(post_ans, post_ans)
            post_ans_correct = post_ans in golden_list

            # mỗi answer chỉ giữ 1 lần
            if norm_ans not in cand_dict:
                cand_dict[norm_ans] = {
                    "is_correct": is_correct,
                    "post_ans": post_ans,
                    "post_ans_correct": post_ans_correct
                }

        result[qidx] = {
            "golden": golden_list,
            "cand": cand_dict
        }

    return result

# ======================
# EXPORT CSV
# ======================
def export_answer_dict_to_csv(answer_dict, out_csv):
    rows = []

    for qidx, info in answer_dict.items():
        golden_str = " | ".join(info["golden"])

        for ans, detail in info["cand"].items():
            rows.append({
                "qidx": qidx,
                "golden": golden_str,
                "answer": ans,
                "post ans": detail["post_ans"],
                "is_correct": detail["is_correct"],
                "post_ans_correct": detail["post_ans_correct"]
            })

    df = pd.DataFrame(
        rows,
        columns=[
            "qidx",
            "golden",
            "answer",
            "post ans",
            "is_correct",
            "post_ans_correct"
        ]
    )
    df.to_csv(out_csv, index=False, encoding="utf-8")

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    JSON_FOLDER = "medhop_results_no_mcq_TFIDF300/all_ans"
    OUTPUT_CSV = "evaluate/answer_level_accuracy.csv"

    # ===== LOAD EXTERNAL DATA =====
    with open("data/all_answer_golden_56.json", "r", encoding="utf-8") as f:
        dict_golden = json.load(f)

    with open("wiki_heading_mapping.json", "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    answer_dict = build_answer_dict(
        json_folder=JSON_FOLDER,
        dict_golden=dict_golden,
        mapping_data=mapping_data
    )

    # optional debug dump
    with open("evaluate/answer_dict.json", "w", encoding="utf-8") as f:
        json.dump(answer_dict, f, indent=2, ensure_ascii=False)

    export_answer_dict_to_csv(answer_dict, OUTPUT_CSV)

    print(f"[DONE] Saved CSV to {OUTPUT_CSV}")
