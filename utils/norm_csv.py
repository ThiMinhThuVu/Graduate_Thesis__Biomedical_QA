import os
import json
import csv

# filepath: e:\UETQuintet\convert_json_to_csv.py
def json_to_csv(input_folder, output_csv):
    # Define the columns for the CSV
    csv_columns = [
        "QIDX", 
        "hop",
        "question", 
        "search_keyword", 
        "key_entity",
        "long_answer", 
        "short_answer", 
        "retrieval_information", 
        "prefix", 
        "name", 
        "suffix", 
        "normalized_verify", 
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
                        # Extract the required fields
                        # row = {
                        #     "QIDX": data.get("QIDX", ""),
                        #     "original_question": data.get("original_question", ""),
                        #     "long_answer": data.get("final_answer", {}).get("long_answer", ""),
                        #     "short_answer": data.get("final_answer", {}).get("short_answer", ""),
                        #     "retrieval_information": data.get("final_answer", {}).get("retrieval_information", ""),
                        #     "prefix": data.get("final_answer", {}).get("prefix", ""),
                        #     "name": data.get("final_answer", {}).get("name", ""),
                        #     "suffix": data.get("final_answer", {}).get("suffix", ""),
                        #     "normalized_full_name": data.get("final_answer", {}).get("normalized_ans", {}).get("full_name", ""),
                        #     "normalized_name": data.get("final_answer", {}).get("normalized_ans", {}).get("name", ""),
                        #     "normalized_prefix": data.get("final_answer", {}).get("normalized_ans", {}).get("prefix", ""),
                        #     "normalized_suffix": data.get("final_answer", {}).get("normalized_ans", {}).get("suffix", ""),
                        # }
                        data_final = data.get("final_answer", {})
                        qidx = data.get("QIDX", "")
                        # print(qidx)
                        path_i = str(len(data["1"])-2)
                        if "1" in data["1"][str(path_i)]:
                            hop = 1
                            data = data["1"][str(path_i)]["1"]
                        else:
                            hop = 0
                            data = data["1"][str(path_i)]["0"]
                        # row = {
                        #     "QIDX": qidx,
                        #     "hop": hop,
                        #     "question": data.get("question", ""),
                        #     "search_keyword": data.get("search_keyword", ""),
                        #     "key_entity": data.get("key_entity", ""),
                        #     "long_answer": data.get("long_answer", ""),
                        #     "short_answer": data.get("short_answer", ""),
                        #     "prefix": data.get("prefix", ""),
                        #     "name": data.get("name", ""),
                        #     "suffix": data.get("suffix", ""),
                        #     "retrieval_information": data.get("retrieval_information", ""),                            
                        #     "normalized_full_name": data_final.get("normalized_ans", {}).get("full_name", ""),
                        #     "normalized_name": data_final.get("normalized_ans", {}).get("name", ""),
                        #     "normalized_prefix": data_final.get("normalized_ans", {}).get("prefix", ""),
                        #     "normalized_suffix": data_final.get("normalized_ans", {}).get("suffix", ""),
                        # }
                        if data_final['normalized_v2'] is None:
                            verify = None
                        elif "normalized_v2" in data_final and len(data_final["normalized_v2"]['verify']) > 0:
                            verify = data_final["normalized_v2"]['verify'][0]
                        else:
                            verify = None
                        row = {
                            "QIDX": qidx,
                            "hop": hop,
                            "question": data.get("question", ""),
                            "search_keyword": data.get("search_keyword", ""),
                            "key_entity": data.get("key_entity", ""),
                            "long_answer": data.get("long_answer", ""),
                            "short_answer": data.get("short_answer", ""),
                            "prefix": data.get("prefix", ""),
                            "name": data.get("name", ""),
                            "suffix": data.get("suffix", ""),
                            "retrieval_information": data.get("retrieval_information", ""),                            
                            "normalized_verify": verify,
                            # "normalized_parent": data_final.get("normalized_v2", {}).get("name", ""),
                            # "normalized_prefix": data_final.get("normalized_v2", {}).get("prefix", ""),
                            # "normalized_suffix": data_final.get("normalized_v2", {}).get("suffix", ""),
                        }
                        writer.writerow(row)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {file_path}")

# Specify the input folder and output CSV file
input_folder = r"e:\UETQuintet\wiki_inference_results\all_ans"
output_csv = r"e:\UETQuintet\infer_output.csv"

# Run the conversion
json_to_csv(input_folder, output_csv)