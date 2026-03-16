import os
import json
import pandas as pd
import wikipedia 
import sys
from tqdm import tqdm
from src.gpt_request import request_api
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_normalization_prompt(question, short_answer, context, wiki_titles):
    system_content = f"""You are an expert in normalizing medical short answers for Exact Match (EM) evaluation.
    Your goal is to extract the precise value, coordinate, or entity that directly answers the question, stripping away all conversational context and descriptive padding.

    Follow this Step-by-Step Normalization Process STRICTLY:

    Step 1: STRICT REDUNDANCY CLEANING & CORE EXTRACTION
    - Delete all conversational lead-in phrases, articles (a, an, the), and connecting words (on, in, at, of, is, located).
    - Remove subjective adjectives (e.g., "common", "severe") but KEEP clinical modifiers (e.g., "Type 2" in "Diabetes Type 2").
    - Remove any repetition of the entity already mentioned in the question.
    - Reduce the answer to its most "atomic" noun, value, or coordinate.

    Step 2: PARENTHESES AND SYMBOL EXTRACTION (HIGH PRIORITY)
    - If the answer follows the format "Full Name (Abbreviation)" or "Entity Name (Symbol)", extract ONLY the content inside the parentheses. 
    Example: "Huntingtin gene (HTT)" -> "HTT".
    - If the answer contains both a verbal description and a specific alphanumeric coordinate/symbol, retain ONLY the coordinate/symbol.

    Step 3: ENTITY TYPE PRESERVATION
    - Do NOT remove words that are part of the formal, canonical name of a medical condition, gene, or organism (e.g., keep "disease" in "Alzheimer's disease", "syndrome" in "Down syndrome", or "virus" in "Influenza virus").

    Step 4: TEXT STANDARDIZATION (GREEK LETTERS)
    - Convert Greek letters to their English word equivalents (e.g., "α" to "alpha", "β" to "beta").

    Step 5: SYMBOL/ABBREVIATION PRIORITY
    - Prefer abbreviations or symbols over full names unless the question explicitly asks for the full name of a human or entity.
    - This rule must NOT override Step 4 for standalone Greek letters.

    Step 6: ENTITY REFERENCE MATCHING (OPTIONAL ALIGNMENT, NOT OVERRIDE)
    You are provided with a list of entity titles (Wiki Titles). Use them ONLY if they are strictly better normalized forms.

        Decision priority order:
        1. The normalized original short answer (after Steps 1–5) is the DEFAULT output.
        2. Replace it with a Wiki Title ONLY if:
        - It is the exact same entity but cleaner / more canonical.
        - It does NOT violate any previous steps.
        - It does NOT expand abbreviations or remove required symbols.
        - It does NOT delete necessary modifiers that are part of the entity's formal name.
        3. If original answer is plural and grammatically correct, keep plural even if Wiki Title is singular.
        4. Wiki Titles MUST NOT override symbol extraction, abbreviation preference, or coordinate preservation.

    GLOBAL CONSTRAINTS:
    - DO NOT introduce any new words, EXCEPT when converting Greek letters (Step 4) or mapping to a provided Wiki Title (Step 5).
    - DO NOT change the spelling of alphanumeric loci coordinates, symbols, or formulas.

    Output Format:
    Provide a strictly valid JSON object containing the key "final_answer".
    """

    user_content = f"""
    Task: Normalize the medical short answer based on the instructions.

    Input Data:
    - Question: "{question}"
    - List of entities: {wiki_titles}
    - Context: "{context}"
    - Original Short Answer: "{short_answer}"

    Provide a strictly valid JSON object. Example:
    {{
        "final_answer": "your_normalized_answer_here"
    }}
    """

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    return messages

def normalize_short_answer(question, short_answer, context, wiki_titles):
    messages = load_normalization_prompt(question, short_answer, context, wiki_titles)
    response = request_api(messages, model='o3-mini')
    # print("Raw GPT response:")
    # print(response)  # <--- in ra đây để xem GPT trả về gì
    # print("-" * 80)

    try: 
        normalized_data = json.loads(response)
        normalized_short_answer = normalized_data["final_answer"]
        return normalized_short_answer
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print("Error parsing response:", e)
        return None  # Fallback to original short answer if parsing fails

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
    submission = load_json('miss_lite_llmpost.json')
    output_file = f"miss_id_lite_cand.json"
    # golden_data = load_json("data/all_answer_golden_56_update_v5.json")
    # metadata = load_json("data/metadata_medhop.json")
    result_dict = {}

    if os.path.exists(output_file):
        result_dict = load_json(output_file)
    else:
        result_dict = {}

    for qidx in tqdm(submission):
        # if qidx not in ["798687"]:
        #     continue

        question = str(submission[qidx]["question"])
        ans = str(submission[qidx]["short_answer"]).lower()
        context = str(submission[qidx]["long_answer"])

        # if qidx in result_dict:
        #     print(f"Skipping already processed QIDX: {qidx}")
        #     continue

        if ans in ['yes', 'no']:
            continue

        wiki_titles = wikipedia.search(ans)
        wiki_titles = [wiki_title.lower() for wiki_title in wiki_titles]

        final_ans = normalize_short_answer(question=question, short_answer=ans, context=context, wiki_titles=wiki_titles)
        result_dict[qidx]['llm_post'] = final_ans
        
        save_json(result_dict, output_file)

    # print(f"Số câu YN đúng: {acc_yn}, Số câu YN: {total_yn}")
    # print(f"Số câu WH đúng: {acc_wh}, Sô câu WH: {total_wh}")