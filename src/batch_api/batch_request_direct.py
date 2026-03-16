from openai import OpenAI
# Giả sử client đã được khởi tạo đúng trong src.api_key
from src.api_key import client 

# 1. Upload file lên hệ thống của OpenAI
file_response = client.files.create(
    file=open("post_llm_lite/Model_Lite/submission_lite_model_10000q_fix_duplicate.jsonl", "rb"),
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
      "description": "POST LLM LITE MODEL"
    }
)

print(f"🚀 Batch Job ID: {batch_job.id}")
print(f"Status: {batch_job.status}")

# (base) PS E:\MAS_UETQuintet> python -m src.batch_api.batch_request_direct
# ✅ Input file ID: file-HaLiJiX31ebNzRQJSoa1rL
# 🚀 Batch Job ID: batch_69a47a4a3efc8190a5fc1ffcb7576277
# Status: validating

#FAIL l1
# (base) PS E:\MAS_UETQuintet> python -m src.batch_api.batch_request_direct
# ✅ Input file ID: file-LVrNEZZY7FnHu9xNM9An9i
# 🚀 Batch Job ID: batch_69a4397de5f88190a3128bf12fbef058
# Status: validating
# (base) PS E:\MAS_UETQuintet> 


# BATCH_ID = "batch_69a4397de5f88190a3128bf12fbef058"   # <-- paste batch_id vào đây

# batch = client.batches.retrieve(BATCH_ID)
# print("Batch status:", batch.status)

# if batch.status != "completed":
#     raise RuntimeError("Batch chưa completed")

# # Download output
# content = client.files.content(batch.output_file_id)

# content.write_to_file(r"src\batch_api\output\zeroshot_o3_v2_3500_9000.jsonl")

# print("Downloaded src\batch_api\output\zeroshot_o3_v2_3500_9000.jsonl")