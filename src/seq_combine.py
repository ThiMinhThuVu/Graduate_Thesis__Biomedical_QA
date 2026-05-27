import os
import json
from src.gpt_request import request_api, request_api_o3_mini
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def prompt_seq_combine(target_answer_type, original_question, context):
    core_instructions = """
        ### INSTRUCTIONS:
        1. **Deconstruct the Question:**
        - Identify the core relationship (e.g., Disease <-> Gene).
        - Identify ALL specific constraints/modifiers (e.g., "with symptom X", "in elderly patients", "specifically for type 2").
        
        2. **Verify Full Coverage (CRITICAL STEP):**
        - Check if the Context covers **every single constraint** found in the question.
        - **Rule of Specificity:** If the question is specific (e.g., "Disease with Symptom X") but the context is general (only mentions "Disease" without confirming/denying Symptom X), you MUST conclude "Insufficient information".
        - Do NOT assume general facts apply to specific conditions unless explicitly stated.
    """

    if target_answer_type == "correctness":
        prompt = f"""You are an expert in logical synthesis and reasoning. 
        ### TASK: 
        Analyze the provided information to determine the correct answer to the original question.

        {core_instructions}

        3. **Determine the Label:**
        - **"yes"**: Context explicitly supports the whole question including all specific constraints.
        - **"no"**: Context explicitly contradicts any part of the question.
        - **"insufficient information"**: Context mentions the entities but misses one or more specific constraints (e.g., context lacks info about the specific symptom, dosage, or population mentioned in the question).

        ## Constraints
        - MUST: Provide ONLY the raw JSON object.
        - MUST NOT: Include any commentary outside the JSON.
        - MUST NOT: Use Markdown formatting or code blocks (like ```json).

        ## Information
        Question: {original_question}
        Context: {context}

        ## Output Format (JSON)
        {{
            "short_answer": "yes", "no", or "insufficient information"
            "long_answer": "Explanation: [Provide a detailed explanation. If short_answer is 'insufficient information', explicitly state which constraint/modifier was missing from the context.]",
        }}
        """
    else:
        prompt = f"""You are an expert in logical synthesis and reasoning. 
        ### TASK: 
        Analyze the provided information to determine the correct answer to the original question.

        {core_instructions}

        3. **Determine the Answer:**
        - **"short_answer"**: If the context fully supports the question, provide the specific answer according to the target_answer_type (e.g., a specific gene, a number, a yes/no).
        - **"insufficient information"**: Context mentions the entities but misses one or more specific constraints (e.g., context lacks info about the specific symptom, dosage, or population).

        ## Rules for Short Answer Formats
        - **Count questions** ("how many"): Return ONLY a number (e.g., "10", "ten")
        - **Time/Duration questions**: Return a concise time value (e.g., "3 days")
        - **Entity questions**: Return the full noun phrase naming the entity (e.g., "chromosome X")
        - **Organism questions**: Return organism type (e.g., "bacterium")
        - **Insufficient Info**: If the information is missing, return exactly "Insufficient information"

        ## Constraints
        - MUST: Provide ONLY the raw JSON object.
        - MUST: Your short_answer MUST be MAX 3 words.
        - MUST NOT: Include any commentary outside the JSON.
        - MUST NOT: Use Markdown formatting or code blocks (like ```json).
        - MUST NOT: Use abbreviations for entities if the full name is available.
        - Target Match: The short_answer MUST align with the provided target_answer_type: {target_answer_type}.

        ## Information
        Question: {original_question}
        Context: {context}

        ## Output Format (JSON)
        {{
            "short_answer": "1-3 word answer strictly matching the {target_answer_type} requirement", or "insufficient information"
            "long_answer": "detailed explanation based on context",
        }}
        """
    return prompt

def collect_context(data):
    evidences = []
    for step in data["1"]["0"]:
        evidences.append({
            "question": data["1"]["0"][step]["question"],
            "short_answer": data["1"]["0"][step]["final_answer"]["short_answer"],
            "long_answer": data["1"]["0"][step]["final_answer"]["long_answer"]
        })
    return evidences

def gen_combine_seq_answer(target_answer_type, original_question, context, model='o3-mini'):
    prompt = prompt_seq_combine(target_answer_type, original_question, context)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "You are a biomedical expert. Your task is to answer a question base on ONLY the provided context"}
    ]

    answer = request_api(messages, model)
    return json.loads(answer)

if __name__ == "__main__":
    RESULT_DIR = "SEQUENCE_LITE_1000_medhop_results/all_ans"
    output_file = "SEQUENCE_LITE_1000_medhop_results/seq_lite_1000_combine.json"
    # RESULT_DIR = "E:/MAS_UETQuintet/ANNQ_SEQUENCE_LITE_1000_medhop_results/SEQUENCE_LITE_1000_medhop_results/all_ans" #"SEQ_medhop_results/all_ans"
    # output_file = "SEQ_medhop_results/seq_combine_zeroshot.json"
    # output_file = r"SEQUENCE_9000_medhop_results\seq_combine[0_9000]_miss_qidx_375532.json"
    # output_file = "ANNQ_SEQUENCE_LITE_1000_medhop_results/SEQUENCE_LITE_1000_medhop_results/seq_combine_zeroshot.json"
    golden_data = load_json("data/all_answer_golden_56_update.json")
    metadata = load_json(r'data/metadata_medhop.json')
    
    if os.path.exists(output_file):
        dict_results = load_json(output_file)
    else:
        dict_results = {}
    cnt = 0
    
    # for file_name in tqdm(os.listdir(RESULT_DIR)):
    #     filepath = os.path.join(RESULT_DIR, file_name)
    for qidx in tqdm(list(metadata.keys())):
        # data = load_json(filepath)
        # qidx = data["QIDX"]
        # type_q = data["type_q"]
        if qidx in dict_results:
            continue
        item = metadata[qidx]
        if item['type_q'] != "sequence":
            continue
        
        filepath = f"{RESULT_DIR}/R4_step_3b_all_ans_{qidx}.json"
        if not os.path.exists(filepath):
            continue
        data = load_json(filepath)
        target_ans_type = item['target_answer_type']
        cnt += 1
        # golden = golden_data[qidx]["True"]
        # if len(golden) == 0:
        #     continue

        # if qidx not in ["281682", "362372", "455108", "521164", "648793", "728328", "773528", "849453"]:
        #     continue

        original_question = item["original question"]
        context = collect_context(data)
        # target_ans_type = 'chromosome' # metadata[qidx]["target_answer_type"] #original question

        final_answer = gen_combine_seq_answer(target_ans_type, original_question, context)
        dict_results[qidx] = final_answer
        print(final_answer)
        
        save_json(dict_results,output_file)
        
