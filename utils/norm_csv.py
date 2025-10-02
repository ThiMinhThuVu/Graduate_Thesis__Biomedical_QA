import os
import json
import csv

# filepath: e:\UETQuintet\convert_json_to_csv.py
def json_to_csv(input_folder, output_csv):
    # Define the columns for the CSV
    csv_columns = [
        "QIDX", 
        "original_question", 
        "long_answer", 
        "short_answer", 
        "retrieval_information", 
        "prefix", 
        "name", 
        "suffix", 
        "normalized_full_name", 
        "normalized_name", 
        "normalized_prefix", 
        "normalized_suffix"
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
                        # Extract the required fields
                        row = {
                            "QIDX": data.get("QIDX", ""),
                            "original_question": data.get("original_question", ""),
                            "long_answer": data.get("final_answer", {}).get("long_answer", ""),
                            "short_answer": data.get("final_answer", {}).get("short_answer", ""),
                            "retrieval_information": data.get("final_answer", {}).get("retrieval_information", ""),
                            "prefix": data.get("final_answer", {}).get("prefix", ""),
                            "name": data.get("final_answer", {}).get("name", ""),
                            "suffix": data.get("final_answer", {}).get("suffix", ""),
                            "normalized_full_name": data.get("final_answer", {}).get("normalized_ans", {}).get("full_name", ""),
                            "normalized_name": data.get("final_answer", {}).get("normalized_ans", {}).get("name", ""),
                            "normalized_prefix": data.get("final_answer", {}).get("normalized_ans", {}).get("prefix", ""),
                            "normalized_suffix": data.get("final_answer", {}).get("normalized_ans", {}).get("suffix", ""),
                        }
                        writer.writerow(row)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {file_path}")

# Specify the input folder and output CSV file
input_folder = r"e:\UETQuintet\wiki_inference_results\all_ans"
output_csv = r"e:\UETQuintet\infer_output.csv"

# Run the conversion
json_to_csv(input_folder, output_csv)