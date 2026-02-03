import json
import csv
import time
import re
from pathlib import Path
from typing import Optional, Dict, List

from prompt_target_ans_type import load_batch_target_answer_type_prompt
from src.gpt_request import request_api


# Configuration
TEST_FILE = Path("data/raw/medhop/test.json")
SHRUNK_FILE = Path("data/raw/medhop/shrunk_mapping.json")
OUTPUT_FILE = Path("data/raw/medhop/target_answer_types_batch.csv")
MODEL = "gpt-4o-mini"
TEMPERATURE = 0
BATCH_SIZE = 5
BATCH_DELAY = 1.0 

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
# cache_class = load_json("data_target_ans_type.json")  

def load_data() -> Dict[str, str]:
    """Load and merge data from test.json and shrunk_mapping.json."""
    print(f"Loading test data from {TEST_FILE}...")
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    print(f"Loading shrunk data from {SHRUNK_FILE}...")
    try:
        with open(SHRUNK_FILE, 'r', encoding='utf-8') as f:
            shrunk_data = json.load(f)
    except FileNotFoundError:
        print("Shrunk mapping file not found. Using only test data.")
        shrunk_data = {}

    merged_data = {}
    shrunk_count = 0
    test_count = 0
    
    for q_id, q_info in test_data.items():
        # Check if ID exists in shrunk_data
        # Note: input keys in test.json are strings.
        if q_id in shrunk_data:
            merged_data[q_id] = shrunk_data[q_id]
            shrunk_count += 1
        else:
            merged_data[q_id] = q_info['Question']
            test_count += 1
            
    print(f"Total samples: {len(merged_data)}")
    print(f"  - From shrunk mapping: {shrunk_count}")
    print(f"  - From original test: {test_count}")
    
    return merged_data


def parse_batch_response(response: str) -> List[Dict]:
    """Parse the JSON list response from the LLM."""
    if not response:
        return []

    try:
        # Try direct parsing
        data = json.loads(response)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
        
    # Extract JSON form code blocks
    json_match = re.search(r'```(?:json)?\s*\n?(\[.*?\])\s*\n?```', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []


def process_batch(batch_questions: Dict[str, str]) -> List[Dict]:
    """Process a batch of questions."""
    messages = load_batch_target_answer_type_prompt(batch_questions)
    
    try:
        response = request_api(messages, MODEL, TEMPERATURE)
        parsed_results = parse_batch_response(response)
        
        # Map back to ensure we have results for all IDs, handling missing/malformed output
        results = []
        parsed_map = {item.get('id'): item.get('target_answer_type') for item in parsed_results if isinstance(item, dict)}
        
        for q_id, question in batch_questions.items():
            ans_type = parsed_map.get(q_id)
            status = "success" if ans_type else "parse_error"
            results.append({
                "id": q_id,
                "question": question,
                "target_answer_type": ans_type,
                "status": status,
                "raw_response": response if status == "parse_error" else None # Only save raw response on error to save space? Or always?
            })
            
        return results

    except Exception as e:
        print(f"Error processing batch: {e}")
        # Return error results for this batch
        return [{
            "id": q_id,
            "question": question,
            "target_answer_type": None,
            "status": "api_error",
            "raw_response": str(e)
        } for q_id, question in batch_questions.items()]


def save_results(results: list, output_path: Path):
    """Save results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ["id", "question", "target_answer_type", "status", "raw_response"]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {output_path}")


def main():
    print("=" * 60)
    print("Batch Target Answer Type Identification Pipeline")
    print("=" * 60)
    
    # 1. Load Data
    data = load_data()
    all_ids = list(data.keys())
    total_samples = len(all_ids)
    
    all_results = []
    
    # 2. Process in Batches
    print(f"\nProcessing {total_samples} samples in batches of {BATCH_SIZE}...")
    
    for i in range(0, total_samples, BATCH_SIZE):
        batch_ids = all_ids[i:i + BATCH_SIZE]
        batch_questions = {qid: data[qid] for qid in batch_ids}
        
        print(f"Processing batch {i // BATCH_SIZE + 1}/{(total_samples + BATCH_SIZE - 1) // BATCH_SIZE} (IDs: {batch_ids[0]}...)...", end=" ")
        
        batch_results = process_batch(batch_questions)
        all_results.extend(batch_results)
        
        # Check success rate of this batch
        successes = sum(1 for r in batch_results if r['status'] == 'success')
        print(f"Done ({successes}/{len(batch_ids)} success)")
        
        time.sleep(BATCH_DELAY)
        
    # 3. Save Results
    print("-" * 60)
    save_results(all_results, OUTPUT_FILE)
    
    # Summary
    success_count = sum(1 for r in all_results if r["status"] == "success")
    print("\n" + "=" * 60)
    print(f"Summary: {success_count}/{total_samples} successful ({success_count/total_samples*100:.1f}%)")
    print("=" * 60)

def main_sequence():
    inputfile = "sequence_last_subquestion.json"
    print("=" * 60)
    print("Batch Target Answer Type Identification Pipeline (Sequence)")
    print("=" * 60)

    # 1. Load Data
    with open(inputfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_ids = list(data.keys())
    total_samples = len(all_ids)
    all_results = []

    print(f"\nProcessing {total_samples} sequence samples in batches of {BATCH_SIZE}...")
    for i in range(0, total_samples, BATCH_SIZE):
        batch_ids = all_ids[i:i + BATCH_SIZE]
        # batch_questions: {qid: subquestion}
        batch_questions = {qid: data[qid]['last_subquestion'] for qid in batch_ids}

        print(f"Processing batch {i // BATCH_SIZE + 1}/{(total_samples + BATCH_SIZE - 1) // BATCH_SIZE} (IDs: {batch_ids[0]}...)...", end=" ")

        batch_results = process_batch(batch_questions)
        all_results.extend(batch_results)

        # Check success rate of this batch
        successes = sum(1 for r in batch_results if r['status'] == 'success')
        print(f"Done ({successes}/{len(batch_ids)} success)")

        time.sleep(BATCH_DELAY)

    # 3. Save Results
    print("-" * 60)
    # Save to a different output file for sequence
    output_file = Path("data/raw/medhop/target_answer_types_batch_sequence.csv")
    save_results(all_results, output_file)

    # Summary
    success_count = sum(1 for r in all_results if r["status"] == "success")
    print("\n" + "=" * 60)
    print(f"Summary: {success_count}/{total_samples} successful ({success_count/total_samples*100:.1f}%)")
    print("=" * 60)
    
if __name__ == "__main__":
    # main()
    main_sequence()
