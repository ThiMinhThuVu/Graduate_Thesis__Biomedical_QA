from openai import OpenAI
# Giả sử client đã được khởi tạo đúng trong src.api_key
from src.api_key import client 

# # 1. Upload file lên hệ thống của OpenAI
# file_response = client.files.create(
#     file=open("src/batch_api/input/o3_direct_v2_3500_9000.jsonl", "rb"),
#     purpose="batch"
# )

# file_id = file_response.id
# print(f"✅ Input file ID: {file_id}")

# # 2. Tạo Batch Job
# batch_job = client.batches.create(
#     input_file_id=file_id,
#     endpoint="/v1/chat/completions",  # BẮT BUỘC phải là cái này cho Chat
#     completion_window="24h",
#     metadata={
#       "description": "ZERO SHOT DIRECT O3-9000"
#     }
# )

# print(f"🚀 Batch Job ID: {batch_job.id}")
# print(f"Status: {batch_job.status}")

# ZERO DIRECT - 9000
#  Input file ID: file-BuAfhk7FyBHdKidg8aN4VU
# 🚀 Batch Job ID: batch_69a1023f28bc8190881d569636a2edab
# Status: validating

BATCH_ID = "batch_69a351e41c3c8190936ac95a8770ce1c"   # <-- paste batch_id vào đây

batch = client.batches.retrieve(BATCH_ID)
print("Batch status:", batch.status)

if batch.status != "completed":
    raise RuntimeError("Batch chưa completed")

# Download output
content = client.files.content(batch.output_file_id)

content.write_to_file(r"src\batch_api\output\o3_seq_parallel_original_question_9000_DAT.jsonl")

print("Downloaded src\batch_api\output\o3_seq_parallel_original_question_9000_DAT.jsonl")