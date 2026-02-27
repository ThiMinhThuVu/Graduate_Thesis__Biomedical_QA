from openai import OpenAI
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from postprocessing import postprocessing
# from src.gpt_request import request_api
from src.api_key import api_key_openai


client = OpenAI(
    api_key = api_key_openai
)

def request_api_o3(messages, max_completion_tokens=8192):
    response = client.chat.completions.create(
        model="o3",
        messages=messages,
        max_completion_tokens=max_completion_tokens
    )
    return response.choices[0].message.content

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prompt_zero_shot_medhop_wh(question):
    prompt = f"""You are an expert in biomedical and scientific knowledge. Your task is to answer the following question based on established scientific consensus and your internal knowledge.

    ### INSTRUCTIONS:
    1. Analyze the input Question carefully to identify the key biomedical entities (e.g., proteins, genes, drugs, diseases) and their relationships.
    2. Retrieve the most relevant scientific or factual knowledge to determine the correct entity that answers the question.
    3. The short_answer MUST be the specific name of the entity or concept requested.
    4. Always provide the most accurate and well-established answer.
    5. Do NOT hedge or speculate. Provide the best possible answer.

    ### INPUT:
    Question: {question}

    ### OUTPUT FORMAT (Strict JSON):
    {{
        "short_answer": "The specific entity name (1–5 words)",
        "long_answer": "A concise explanation supporting the short_answer, mentioning key scientific facts or mechanisms."
    }}
    """
    return prompt

def prompt_zero_shot_medhop_correctness(question):
    prompt = f"""You are an expert in biomedical and scientific knowledge. Your task is to answer the following question based on established scientific consensus and your internal knowledge.
        ### INSTRUCTIONS:
        1. Analyze the input 'Question'. Identify key biomedical entities (e.g., drugs, proteins, diseases) and their relationships.
        2. Retrieve relevant medical facts and established scientific consensus regarding these entities.
        3. Determine the label based on the following criteria:
        - **"yes"**: The statement is factually correct and supported by established biomedical consensus.
        - **"no"**: The statement is factually incorrect or contradicted by established biomedical consensus.

        ### INPUT DATA:
        Question: {question}

        ### OUTPUT FORMAT (Strict JSON):
        {{  
            "short_answer": "yes" or "no",
            "long_answer": "[Detailed explanation based on the information]"
        }}
        """
    return prompt

# def build_batch_jsonl(output_jsonl):
#     data = load_json(r'E:\MAS_UETQuintet\zero-shot-o3\direct\list_id_miss_o3_zeroshot.json')
        
#     with open(output_jsonl, "w", encoding="utf-8") as out:
#         for i, (qidx, sample) in enumerate(data.items()):
#             question = sample["original question"]
#             if sample['target_answer_type'] == 'correctness':
#                 prompt = prompt_zero_shot_medhop_correctness(question)
#             else:
#                 prompt = prompt_zero_shot_medhop_wh(question)

#             messages = [
#                 {"role": "user", "content": prompt}
#             ]

#             record = {
#                 "custom_id": f"{qidx}", # Fix: Đảm bảo ID là duy nhất cho mỗi dòng
#                 "method": "POST",
#                 "url": "/v1/chat/completions", # Fix: Endpoint đúng cho chat
#                 "body": {
#                     "model": "o3", 
#                     "messages": messages,
#                     "response_format": { "type": "json_object" } # Khuyên dùng để o3 trả về JSON chuẩn
#                 }
#             }

#             out.write(json.dumps(record, ensure_ascii=False) + "\n")

#     print(f"Saved batch input to {output_jsonl}")
    

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
import os
def build_batch_jsonl():
    output_file = "zeroshot-parallel_9000.json"
    data = load_json(r'data\metadata_9000q.json')
    BASE = "DIRECT_9000_medhop_results/all_ans"
    cnt = 0
    evidence = 0
    output = {}
    # CACHE = load_json(r'src\batch_api\output\format\zeroshot_o3_v0_0_3500.json')
    CACHE = {}
    for i, (qidx, sample) in enumerate(data.items()):
        if sample['type_q'] != 'parallel':
            continue
        
        if qidx in CACHE:
            continue
        
        filepath = f"{BASE}/R4_step_3b_all_ans_{qidx}.json"
        if not os.path.exists(filepath):
            continue
        
         
        item_data = load_json(filepath)
        status = item_data['1']['0']['0']['final_answer']['selection_status']
        if status == "selected":
            evidence += 1
            continue
        if item_data['1']['0']['0']['final_answer']['short_answer'] == 'insufficient information':
            cnt += 1
        target_answer_type = item_data['1']['0']['0']['target_answer_type']
        
        question = item_data['original_question']
        # cnt += 1
        if target_answer_type == 'correctness':
            prompt = prompt_zero_shot_medhop_correctness(question)
        else:
            prompt = prompt_zero_shot_medhop_wh(question)

        result_json = request_api_o3([{"role": "user", "content": prompt}])

        result_dict = json.loads(result_json)
        
        output[qidx] = result_dict
        
        save_json(output, output_file)
    save_json(output, output_file)
    print(len(output.keys()))
if __name__ == "__main__":
    build_batch_jsonl()
