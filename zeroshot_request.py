import json
import os
import sys
from openai import OpenAI
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "")))
from postprocessing import postprocessing
from src.api_key import api_key_openai

client = OpenAI(api_key=api_key_openai)

def load_json(path):
    if not os.path.exists(path):
        print(f"Cảnh báo: Không tìm thấy file tại {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def prompt_zero_shot_medhop_wh(question):
    return f"""You are an expert in biomedical and scientific knowledge. Your task is to answer the following question based on established scientific consensus and your internal knowledge.

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
    
def prompt_zero_shot_medhop_yn(question):
    prompt = f"""You are a biomedical expert. Your task is to answer a question base on ONLY the provided context
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

def request_api(messages, model='o3', max_completion_tokens=8192):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens
    )
    return response.choices[0].message.content

def gen_zero_shot_answer_medhop(question, target_answer_type, model='o3'):
    if target_answer_type == "correctness":
        prompt = prompt_zero_shot_medhop_yn(question)
    else:
        prompt = prompt_zero_shot_medhop_wh(question)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "You are a biomedical expert. Your task is to answer a question base on ONLY the provided context"}
    ]

    answer = request_api(messages, model)
    return json.loads(answer)

# def run_zero_shot_direct_9000q():
#     metadata = load_json("data/metadata_9000q.json")
#     cache = load_json("MEDHOP_10K_ZERO_SHOT_CACHE/zeroshot_o3_9000q_direct.json")

#     for qidx in metadata:
#         if qidx in cache:
#             print(f"Already have qidx: {qidx}. Skip")
#         else:
#             question = metadata[qidx]["original question"]
#             type_q = metadata[qidx]["type_q"]
#             target_ans_type = metadata[qidx]["target_answer_type"]

#             if type_q != "direct":
#                 continue

#             final_answer = gen_zero_shot_answer_medhop(question, target_ans_type)
#             cache[qidx] = final_answer
#             save_json(cache, "MEDHOP_10K_ZERO_SHOT_CACHE/direct_1000_backup.json")
#             break

def run_zero_shot_seq_9000q():
    metadata = load_json("data/metadata_9000q.json")
    target_ans_dict = load_json("data/sequence_target_answer_type.json")
    cache_filepath = "experiment_o3_ZEROSHOT/zero_shot_9981q.json"

    if os.path.exists(cache_filepath):
        cache = load_json(cache_filepath)
    else:
        cache = {}
    cnt = 0
    for qidx in tqdm(target_ans_dict):
        if qidx in cache:
            print(f"Skipping QIDX: {qidx}")
            continue

        target_ans_type = target_ans_dict[qidx]
        question = metadata[qidx]["Question"]

        final_answer = gen_zero_shot_answer_medhop(question=question, target_answer_type=target_ans_type)
        cache[qidx] = final_answer

        save_json(cache, cache_filepath)
        cnt += 1


def run_zero_shot_seq_1000q():
    metadata = load_json("data/metadata_medhop.json")
    cache_filepath = "experiment_o3_ZEROSHOT/zero_shot_9981q.json"

    if os.path.exists(cache_filepath):
        cache = load_json(cache_filepath)
    else:
        cache = {}
    cnt = 0
    for qidx in tqdm(list(metadata.keys())):
        if qidx in cache:
            print(f"Skipping QIDX: {qidx}")
            continue
        
        target_ans_type = metadata[qidx]['target_answer_type']
        question = metadata[qidx]["original question"]
        
        final_answer = gen_zero_shot_answer_medhop(question=question, target_answer_type=target_ans_type)
        cache[qidx] = final_answer

        save_json(cache, cache_filepath)
        cnt += 1
        print(cnt, qidx, target_ans_type)
        


def run_zero_shot_parallel_1000q():
    metadata = load_json("data/metadata_medhop.json")
    cache_filepath = "experiment_o3_ZEROSHOT/zero_shot_9981q.json"

    if os.path.exists(cache_filepath):
        cache = load_json(cache_filepath)
    else:
        cache = {}
    cnt = 0
    for qidx in tqdm(list(metadata.keys())):
        if qidx in cache:
            print(f"Skipping QIDX: {qidx}")
            continue
        
        target_ans_type = metadata[qidx]['target_answer_type']
        question = metadata[qidx]["original question"]
        
        final_answer = gen_zero_shot_answer_medhop(question=question, target_answer_type=target_ans_type)
        cache[qidx] = final_answer

        save_json(cache, cache_filepath)
        cnt += 1
        print(cnt, qidx, target_ans_type)
        
        
if __name__ == "__main__":
    # run_zero_shot_seq_9000q()
    run_zero_shot_seq_1000q()