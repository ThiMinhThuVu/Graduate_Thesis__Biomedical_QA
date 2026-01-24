
import json
import os
import pandas as pd
import re
import unicodedata
from utils.util import load_yaml
from classification.classification_run import load_model_classification, classify_question

DASHES = {
    "–": "-",
    "—": "-",
    "−": "-",
}

def remove_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )
    
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"^['\"]|['\"]$", "", text)
    # 1. Unicode normalize
    text = unicodedata.normalize("NFKC", text)
    text = remove_accents(text)
    # 2. Chuẩn hóa dash
    for k, v in DASHES.items():
        text = text.replace(k, v)
    # 3. Xóa (the)
    text = re.sub(r'\(\s*the\s*\)\s*', '', text, flags=re.IGNORECASE)
    # 4. Xóa ngoặc vuông / nhọn
    text = re.sub(r'\[[^]]*\]', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    # 5. Strip ký tự cho phép
    text = re.sub(r'^[\n;!#*\.,]+|[\n;!#*\.,]+$', '', text)
    # 6. Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)
def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

cache_class = load_json("data/important_id.json")

config_path = "config/config.yaml"
configs = load_yaml(config_path)
dataset = configs['run']['dataset']

model_path = configs['classification'][dataset]['model_path']
tokenizer_dir = configs['classification'][dataset]['tokenizer_dir']

model, tokenizer, device = load_model_classification(dataset, model_path, tokenizer_dir)

# Combined output: one file with all relevant columns
all_ans = []
all_ans_dir = "medhop_results/all_ans"
ground_truths = load_json("data/all_answer_golden.json")
labels_map = {'0': 'direct', '1': 'sequence'}
for filename in os.listdir(all_ans_dir):
    id = filename.replace(".json", "").split("_")[-1]
    file_path = os.path.join(all_ans_dir, filename)
    data = load_json(file_path)
    question = data.get('original_question', "")
    type_q, probabilities = classify_question(dataset, model, tokenizer, device, question)

    # --- Candidate normalization changes ---
    norm_original = ""
    norm_normalized = ""
    qi_keys = [k for k in data.keys() if k.isdigit()]
    if qi_keys:
        qi_last_key = sorted(qi_keys, key=int)[-1]
        path_data = data[qi_last_key]
        path_last_keys = list(path_data.keys())[-1]
        step_data = path_data[path_last_keys]
        step_keys = [k for k in step_data.keys() if k.isdigit()]
        if step_keys:
            last_step = sorted(step_keys, key=int)[-1]
            last_step_data = step_data[last_step]
            candidates = last_step_data.get("candidate", [])
            ans_type = last_step_data.get('target_answer_type', '')
            
            if candidates:
                first_cand = candidates[0]['full_data']
                norm_info = first_cand.get("normalization", None)
                if norm_info and norm_info.get("was_normalized") and "changes" in norm_info:
                    # Only take the first change for summary (or join if multiple)
                    change = norm_info["changes"][0]
                    norm_original = change.get("original", "")
                    norm_normalized = change.get("normalized", "")
                else:
                    orig_short = first_cand.get("short_answer", "")
                    norm_original = orig_short
                    # norm_normalized = normalize_text(orig_short)
                    norm_normalized = None

    # --- Answer/ground truth comparison ---
    gt_true = ground_truths[id]['True'] if id in ground_truths else []
    gt_false = ground_truths[id]['False'] if id in ground_truths else []
    pred_short_answer = data.get("final_answer", {}).get("short_answer", "")
    pred_long_answer = data.get("final_answer", {}).get("long_answer", "")
    q_type_v0 = labels_map[str(cache_class[id]['is_sequence'])]
    all_ans.append({
        "id": id, 
        "q_type_v0": q_type_v0,
        "q_type": type_q, 
        "props": probabilities,
        'target_ans_type': ans_type,
        "norm_ans_is_correct": normalize_text(norm_normalized) in gt_true,  
        "original_ans_is_correct": normalize_text(norm_original) in gt_true, 
        "ground_truth_true": gt_true,
        # "ground_truth_false": gt_false,
        "pred_short_answer": pred_short_answer,
        "short_ans_original": norm_original,
        "short_ans_normalized": norm_normalized,
        "pred_long_answer": pred_long_answer,
        
    })

pd.DataFrame(all_ans).to_csv("answer_and_normalization_combined.csv", index=False)
