from openai import OpenAI
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from postprocessing import postprocessing

api_key = "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"
client = OpenAI(
    api_key = api_key
)

def request_api_o3(messages, max_completion_tokens=8192):
    response = client.chat.completions.create(
        model="o3",
        messages=messages,
        max_completion_tokens=max_completion_tokens
    )
    return response.choices[0].message.content
    
def prompt_zero_shot_medhop_wh(question, target_answer_type):
    prompt = f"""You are an expert in biomedical and scientific knowledge. Your task is to answer the following question based on established scientific consensus and your internal knowledge.

    ### INSTRUCTIONS:
    1. Analyze the input Question carefully and identify the key concepts and relationships.
    2. Retrieve the most relevant scientific or factual knowledge.
    3. The expected answer type is: {target_answer_type}.
    4. The short_answer MUST match the expected answer type.
    5. Always provide the most accurate and well-established answer.
    6. Do NOT hedge or speculate. Provide the best possible answer.

    ### INPUT:
    Question: {question}

    ### OUTPUT FORMAT (Strict JSON):
    {{
        "short_answer": "A concise answer (1–5 words) matching the expected type {target_answer_type}",
        "long_answer": "Detailed explanation supporting the short_answer"
    }}
    """
    return prompt


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
GOLDEN_PATH = "data/all_answer_golden_56.json"

mapping_data = load_json("data/mapping_answer_gol.json")
dict_golden = load_json(GOLDEN_PATH)
metadata = load_json('data/metadata_medhop.json')
# input_data = load_json(r'zero_shot_wh_81.json')
# input_data = load_json(r'unselected_full_candidates.json')

tmp_data = load_json(r'unselected_full_candidates.json')
output = load_json('o3_gen_ans_wh_115.json')

# for i, (qidx, question) in enumerate(input_data.items()):
#     if qidx in output:
#         continue
#     print("xu ly: ", qidx)
#     target_answer_type = metadata[qidx]['target_answer_type']
#     prompt = prompt_zero_shot_medhop_wh(question, target_answer_type)
#     result_json = request_api_o3([{"role": "user", "content": prompt}])
#     result_dict = json.loads(result_json)
    
#     output[qidx] = {
#         'o3_result': result_dict
#     } 
#     with open('o3_gen_ans_wh_126.json', 'w', encoding='utf-8') as f:
#         json.dump(output, f, ensure_ascii=False, indent=2)

# with open('o3_gen_ans_wh_81.json', 'w', encoding='utf-8') as f:
#         json.dump(output, f, ensure_ascii=False, indent=2)

# candidates = load_json(r'request_batch\mcq_candidate_wh.json')
# output = {}
# for qidx, cands in candidates.items():
#     question = metadata[qidx]['original question']
#     target_answer_type = metadata[qidx]['target_answer_type']
#     prompt = prompt_zero_shot_medhop_wh(question, target_answer_type)
#     result_json = request_api_o3([{"role": "user", "content": prompt}])
#     result_dict = json.loads(result_json)
    
#     output[qidx] = {
#         'o3_result': result_dict,
#         'candidate': cands
#     } 

# with open('o3_gen_ans_wh_58.json', 'w', encoding='utf-8') as f:
#     json.dump(output, f, ensure_ascii=False, indent=2)
    
cnt = 0
for qidx, item in output.items():

    list_golden = dict_golden[qidx]['True']
    ans = item['o3_result']['short_answer'].lower()
    item['o3_result']['short_answer'] = ans
    if ans in mapping_data:
        ans = mapping_data[ans]
    else:
        ans = postprocessing(ans)
        ans = mapping_data.get(ans, ans)
    
    if ans in list_golden:
        check = True
        cnt += 1
    else:
        check = False
        
    item['golden'] = list_golden
    item['check'] = check
    item['candidate'] = tmp_data.get(qidx, {}).get('candidates')
    
list_id = []
for qidx in output.keys():
    if qidx not in tmp_data:
        list_id.append(qidx)
for qidx in list_id:
    del output[qidx]
        
with open(f'o3_gen_ans_wh_{len(output)}.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(cnt, "/", len(output.keys()))