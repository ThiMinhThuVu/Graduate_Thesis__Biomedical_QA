import csv
import json

input_csv = "wiki_heading_mapping.csv"
output_json = "wiki_heading_mapping.json"

result = {}

with open(input_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Only take rows where Delete != 'x'
        if row.get("Delete", "").strip() != "x":
            key = row["short_ans"].strip()
            result[key] = row["alias_golden"].strip()
              

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Saved: {output_json}")
