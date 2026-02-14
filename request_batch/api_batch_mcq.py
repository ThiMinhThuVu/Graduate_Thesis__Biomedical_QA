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
short_answer: <one answer from choices OR "insufficient information">
long_answer: <chosen answer or "insufficient information">. Reason: <brief explanation based only on the provided reasoning>.
"""

def format_answers(ans_dict):
    """
    Chuyển dict answers thành text gọn cho prompt
    """
    lines = []
    for v in ans_dict:
        lines.append(v['final_text'])
    return "\n".join(lines)

def build_batch_jsonl(output_jsonl):
    data = load_json('mcq_candidate_wh.json')
        
    metadata = load_json(r'data/metadata_medhop.json')

    with open(output_jsonl, "w", encoding="utf-8") as out:
        for i, (qidx, sample) in enumerate(data.items()):
            prompt = PROMPT_TEMPLATE.format(
                question=metadata[qidx]["original question"],
                target_answer_type=metadata[qidx]["target_answer_type"],
                ans=format_answers(sample)
            )

            record = {
                # "qidx": qidx,
                "custom_id": f"qidx",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": "o3",
                    "input": prompt
                }
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved batch input to {output_jsonl}")

if __name__ == "__main__":
    build_batch_jsonl(
        output_jsonl="batch_input_tmp.jsonl"
    )
