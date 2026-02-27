import json
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

import os
import re
        
def extract_references(long_answer):
    if not long_answer:
        return []
    return re.findall(r"\[([^\]]+)\]", long_answer)


def select_final_answer(zeroshot_ans, candidates):
    final_answer = zeroshot_ans
    for label, cand in candidates.items():
        if cand['llm']['short_answer'].strip().lower() == zeroshot_ans['short_answer'].strip().lower():
            final_answer['long_answer'] = cand['llm']['long_answer']
            final_answer['selection_status'] = "selected"
            final_answer['sources'] = extract_references(cand['llm']['long_answer'])
            final_answer['context'] = cand['context']
            
            return final_answer
            
    return final_answer

def build_batch_jsonl(output_jsonl):
    data = load_json(r'data\metadata_9000q.json')
    BASE = "DIRECT_9000_medhop_results/all_ans"
    cnt = 0
    evidence = 0
    CACHE = load_json(r'src\batch_api\output\format\zeroshot_o3_v0_0_3500.json')
    CACHE.update(load_json(r'src\batch_api\output\format\zeroshot_o3_v2_3500_9000.json'))
    print(len(CACHE.keys()))
    
    for i, (qidx, sample) in enumerate(data.items()):
        if sample['type_q'] != 'direct':
            continue
        
        qidx = str(qidx)
        if qidx not in CACHE:
            continue
        
        filepath = f"{BASE}/R4_step_3b_all_ans_{qidx}.json"
        if not os.path.exists(filepath):
            continue
        
        item_data = load_json(filepath)
        
        final_answer = item_data['1']['0']['0']['final_answer']
        status = final_answer['selection_status']
        
        cluster_generated_answer = item_data['1']['0']['0']['cluster_genered_answer']
        if status == "selected":
            evidence += 1
            continue
        else:
            cnt += 1
            zeroshot_answer = {
                "short_answer": CACHE[qidx]['short_answer'],
                "long_answer": CACHE[qidx]['long_answer'],
                "selection_status": "zero-shot_fallback",
                "cluster_score": None
            }
            final_answer = select_final_answer(zeroshot_ans=zeroshot_answer, candidates=cluster_generated_answer)
        
        item_data['1']['0']['0']['final_answer'] = final_answer 
        item_data['1']['final_qi']['long_answer'] = final_answer['long_answer']
        
        item_data['final_answer'] = {
            'short_answer': final_answer['short_answer'],
            'long_answer': final_answer['long_answer']
        }
        cnt += 1
        save_json(item_data, filepath)
        # print(filepath)
    print(cnt)
    print(evidence, cnt + evidence)
    print(f"Saved batch input to {output_jsonl}")
if __name__ == "__main__":
    build_batch_jsonl(
        output_jsonl="src/batch_api/input/o3_direct_v3.jsonl"
    )
