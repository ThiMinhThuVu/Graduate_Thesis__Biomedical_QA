import json
import os 
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def load_json(golden_path):
    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)
    return golden_data

def load_mapping_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    mapping_dict = load_json("wiki_heading_mapping.json")

    for _, row in df.iterrows():
        # chỉ lấy các case sai
        if str(row["post_ans_correct"]).lower() == "false":
            post_ans = str(row["post ans + mapping"]).strip()
            mapping = str(row["Mapping"]).strip()

            if post_ans and mapping:
                mapping_dict[post_ans] = mapping

    with open("mapping_answer_gol.json", "w", encoding='utf-8') as f:
        json.dump(mapping_dict, f, indent=2, ensure_ascii=False)

load_mapping_from_csv(r"evaluate\acc\answer_mapping.csv")