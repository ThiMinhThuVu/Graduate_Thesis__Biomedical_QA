import os
import json
import csv
from utils.util import load_yaml
from classification.classification_run import load_model_classification, classify_question

def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# config_path = "config/config.yaml"
# configs = load_yaml(config_path)
# dataset = configs['run']['dataset']

# model_path = configs['classification'][dataset]['model_path']
# tokenizer_dir = configs['classification'][dataset]['tokenizer_dir']

# model, tokenizer, device = load_model_classification(dataset, model_path, tokenizer_dir)

def save_qidx_list_to_csv(qidx_dict, base_dir="medhop_preprocessing/step1", output_csv="analysis_decompose.csv"):
    folders = ["1", "2", "3", "4"]
    # folders = ["1"]
    results = []
    
    list_id = [        
        # "284704",
        # "373907",
        # "378607",
        
        # "143907",
        # "195446",
        
        # "406222",
        # "521164",
        # "860205",
        # "875070",
        # "894544",
        
        # "192672",
        # "238256",
        # "305681",
        # "326868",
        # "341684",
        # "353225",
        
        # "152701",
        # "214705",
        # "246741",
        # "273077",    
            
        # "446401",
        # "491322",
        # "527304",
        # "852613",
        # "904524",
        # "912403",
        # "954923"
    ]


    for qidx, item in qidx_dict.items():
        # if qidx not in list_id:
        #     continue
        result = {"qidx": qidx}    
        # result = {"qidx": qidx, "Question": item['Question'], "Type_q": item['type_q']}
        for folder in folders:
            folder_path = os.path.join(base_dir, folder)
            found = False
            for filename in os.listdir(folder_path):
                if filename.endswith(".json") and str(qidx) in filename:
                    file_path = os.path.join(folder_path, filename)
                    # if qidx in list_id:
                    #     os.remove(file_path)  
                    #     break
                    
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        data = data.get("0", {})
                        for step, item in data.items():
                            result[f"{folder}_{step}_subq"] = item.get("subquestion", "")
                            result[f"{folder}_{step}_ans_type"] = item.get("target_answer_type", "")

            
        results.append(result)

    # Tìm tất cả các trường xuất hiện trong các result
    all_fieldnames = set()
    for r in results:
        all_fieldnames.update(r.keys())
    fieldnames = list(sorted(all_fieldnames))

    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

dict_id = load_json("data/important_id.json")

# for id, item in dict_id.items():
#     question = item["Question"]
#     type_q, probabilities = classify_question(dataset, model, tokenizer, device, question)
#     item['type_q'] = type_q

# cache_csv = 'answer_and_normalization_combined.csv'
# dict_id = {}
# with open(cache_csv, 'r', encoding='utf-8') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         dict_id[row['id']] = row['question']

save_qidx_list_to_csv(dict_id)