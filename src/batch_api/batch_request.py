from openai import OpenAI
# Giả sử client đã được khởi tạo đúng trong src.api_key
from src.api_key import client 

# 1. Upload file lên hệ thống của OpenAI
file_response = client.files.create(
    file=open("src/batch_api/input/o3_direct_v1.jsonl", "rb"),
    purpose="batch"
)

file_id = file_response.id
print(f"✅ Input file ID: {file_id}")

# 2. Tạo Batch Job
batch_job = client.batches.create(
    input_file_id=file_id,
    endpoint="/v1/chat/completions",  # BẮT BUỘC phải là cái này cho Chat
    completion_window="24h",
    metadata={
      "description": "ZERO SHOT DIRECT O3"
    }
)

print(f"🚀 Batch Job ID: {batch_job.id}")
print(f"Status: {batch_job.status}")

# ZERO DIRECT
# ✅ Input file ID: file-UTjKT5ENh7TtdYZGmo9j44
# 🚀 Batch Job ID: batch_699f5415ba28819088e4b9a52df0892b
# Status: validating

# EXTRACT NER
# ✅ Input file ID: file-4vkxQESCu8WiwwDSCS22Vr
# 🚀 Batch Job ID: batch_699f3c6f2b508190bbd6bac1d5babfa1
# Status: validating

BATCH_ID = "batch_699f5415ba28819088e4b9a52df0892b"   # <-- paste batch_id vào đây

batch = client.batches.retrieve(BATCH_ID)
print("Batch status:", batch.status)

if batch.status != "completed":
    raise RuntimeError("Batch chưa completed")

# Download output
content = client.files.content(batch.output_file_id)
# content.write_to_file(r"src\batch_api\output\ner_llm_9000.jsonl")

# print("Downloaded src\batch_api\output\ner_llm_9000.jsonl")

content.write_to_file(r"src\batch_api\output\zeroshot_o3_v0_9000.jsonl")

print("Downloaded src\batch_api\output\zeroshot_o3_v0_9000.jsonl")