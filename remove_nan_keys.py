import json

def remove_nan_keys(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Remove keys with value == 'nan' (as string)
    cleaned = {k: v for k, v in data.items() if v != 'nan'}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

# Example usage:
remove_nan_keys("data/mapping_answer_gol_old.json", "data/mapping_answer_gol.json")
