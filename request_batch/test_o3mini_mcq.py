from src.gpt_request import request_api
import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

PROMPT_TEMPLATE = """You are a biomedical expert.

Answer the multiple-choice question by reasoning ONLY over the provided answer choices.
Each answer choice includes its own reasoning/evidence.

Question: {question}
Target answer type: {target_answer_type}
Answer choices (with reasoning): {ans}

Instructions:
- Analyze and compare the reasoning provided for each answer choice.
- Select the SINGLE best answer whose reasoning is most consistent with the question and matches the target answer type.
- Do NOT introduce new entities or use external knowledge.
- If none of the answer choices provide sufficient or relevant reasoning, return "insufficient information".

Output format (strict):
Return your answer as a valid JSON object with two keys:
    "short_answer": <one answer from choices OR "insufficient information">,
    "long_answer": <chosen answer or "insufficient information">. Reason: <brief explanation based only on the provided reasoning>
"""

def format_answers(ans_dict):
    """
    Chuyển dict answers thành text gọn cho prompt
    """
    lines = []
    for v in ans_dict:
        lines.append(v['final_text'])
    return "\n".join(lines)

input_data = load_json(r'mcq_candidate_wh.json')
metadata = load_json(r'data/metadata_medhop.json')
output = {}

for qidx, sample in input_data.items():
    prompt = PROMPT_TEMPLATE.format(
        question=metadata[qidx]["original question"],
        target_answer_type=metadata[qidx]["target_answer_type"],
        ans=format_answers(sample)  # Use the correct key for answer choices
    )
    result_json = request_api([{"role": "user", "content": prompt}], model="o3-mini")
    result_dict = json.loads(result_json)
    
    output[qidx] = {
        'mcq': result_dict,
        'candidate': sample
    } 


with open('o3_mini_mcq_wh_58.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)