from openai import OpenAI
from src.api_key import client
import json
# file = client.files.create(
#     file=open("request_batch/batch_input_58.jsonl", "rb"),
#     purpose="batch"
# )

# print("Input file ID:", file.id)

# batch = client.batches.create(
#     input_file_id=file.id,
#     endpoint="/v1/responses",
#     completion_window="24h"
# )

# print("Batch ID:", batch.id)


#Input file ID: file-Jd7s9XtbRQMuRYYgtRbK7t
# Batch ID: batch_698e3b97777c8190b222e6b3a61c2a3b
# validating


# BATCH_ID = "batch_698e3b97777c8190b222e6b3a61c2a3b"   # <-- paste batch_id vào đây

# batch = client.batches.retrieve(BATCH_ID)
# print("Batch status:", batch.status)

# if batch.status != "completed":
#     raise RuntimeError("Batch chưa completed")

# # Download output
# content = client.files.content(batch.output_file_id)
# content.write_to_file("batch_output_58.jsonl")

# print("Downloaded batch_output_58.jsonl")

# Parse
results = {}
with open("batch_output_58.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        try:
            outputs = obj["response"]["body"].get("output", [])
            answer_text = None
            for out in outputs:
                if out.get("type") == "message":
                    content = out.get("content", [])
                    if content and isinstance(content, list):
                        answer_text = content[0].get("text")
                        break
            # Parse answer_text into short_answer and long_answer
            short_answer = None
            long_answer = None
            if answer_text:
                for part in answer_text.split("\n"):
                    if part.startswith("short_answer:"):
                        short_answer = part[len("short_answer:"):].strip()
                    elif part.startswith("final_text:"):
                        long_answer = part[len("final_text:"):].strip()
            results[obj["custom_id"]] = {
                "short_answer": short_answer,
                "long_answer": long_answer
            }
        except Exception as e:
            results[obj["custom_id"]] = {"error": f"Error parsing: {e}"}

print("Total results:", len(results))

# Save results to a JSON file
with open("batch_output_58_results_split.json", "w", encoding="utf-8") as out_f:
    json.dump(results, out_f, ensure_ascii=False, indent=2)
print("Results saved to batch_output_58_results_split.json")