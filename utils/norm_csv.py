import os
import json
import csv

# filepath: e:\UETQuintet\convert_json_to_csv.py
def json_to_csv(input_folder, output_csv):
    # Define the columns for the CSV
    cnt = 0

    csv_columns = [
        "QIDX", 
        "hop",
        "question", 
        "search_keyword", 
        "key_entity",
        "long_answer", 
        "short_answer", 
        "retrieval_information", 
        # "prefix", 
        # "name", 
        # "suffix", 
        # "normalized_verify", 
        # "normalized_name", 
        # "normalized_prefix", 
        # "normalized_suffix"
    ]

    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        # Iterate through all JSON files in the folder
        for filename in os.listdir(input_folder):
            if filename.endswith('.json'):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as jsonfile:
                    try:
                        data = json.load(jsonfile)
                        
                        data_final = data.get("final_answer", {})
                        qidx = data.get("QIDX", "")
                        
                        short_ans = data_final.get("answer_oldid", "")
                        if short_ans == "" or short_ans is None:
                            cnt += 1
                            short_ans = data_final.get("short_answer", "")
                        
                        row = {
                            "QIDX": qidx,
                            # "hop": hop,
                            "question": data.get("question", ""),
                            "search_keyword": data.get("search_keyword", ""),
                            "key_entity": data.get("key_entity", ""),
                            "long_answer": data.get("long_answer", ""),
                            "short_answer": short_ans,
                        }
                        writer.writerow(row)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {file_path}")
    print(cnt)

# Specify the input folder and output CSV file
input_folder = r"e:\UETQuintet\wiki_inference_results\results"
output_csv = r"e:\UETQuintet\infer_output_v2.csv"

# Run the conversion
json_to_csv(input_folder, output_csv)
