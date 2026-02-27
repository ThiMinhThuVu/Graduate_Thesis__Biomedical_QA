import json
from postprocessing import postprocessing

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


# def select_final_answer(zeroshot_ans, candidates):
#     final_answer = zeroshot_ans
#     for label, cand in candidates.items():
#         if cand['llm']['short_answer'].strip().lower() == zeroshot_ans['short_answer'].strip().lower():
#             final_answer['long_answer'] = cand['llm']['long_answer']
#             final_answer['selection_status'] = "selected"
#             final_answer['sources'] = extract_references(cand['llm']['long_answer'])
#             final_answer['context'] = cand['context']
            
#             return final_answer
            
#     return final_answer

def select_final_answer(zeroshot_ans, candidates):
    final_res = zeroshot_ans.copy()
    
    zs_short = zeroshot_ans['short_answer'].strip().lower()

    for label, cand in candidates.items():
        cand_short = cand['llm']['short_answer'].strip().lower()
        
        if cand_short == zs_short:
            final_res.update({
                'long_answer': cand['llm']['long_answer'],
                'selection_status': "selected_from_candidate", # Đổi tên để dễ debug
                'sources': extract_references(cand['llm']['long_answer']),
                'context': cand['context'],
                'cluster_score': None
            })
            return final_res
            
    return final_res

def build_batch_jsonl(output_jsonl):
    data = load_json(r'data\metadata_medhop.json')
    # BASE = "DIRECT_9000_medhop_results/all_ans"
    BASE = "data/ALL_ROUND/1000/medhop_results_gg_direct/all_ans"
    PATH_MAPPING = r'data\mapping_answer_gol_add_seq.json'
    mapping_data = load_json(PATH_MAPPING)
    CACHE = load_json(r'data\cache\O3_zeroshot_ans_cache.json')
    cnt = 0
    evidence = 0
    # CACHE = load_json(r'src\batch_api\output\format\zeroshot_o3_v0_0_3500.json')
    # CACHE.update(load_json(r'src\batch_api\output\format\zeroshot_o3_v2_3500_9000.json'))
    # print(len(CACHE.keys()))
    
    for i, (qidx, sample) in enumerate(data.items()):
        if sample['type_q'] != 'direct':
            continue
        
        qidx = str(qidx)
        if qidx not in CACHE:
            continue
        
        filepath = f"{BASE}/R4_step_3b_all_ans_{qidx}.json"
        if not os.path.exists(filepath):
            continue
        
        sample_data = load_json(filepath)
        
        for qi in ['1', '2', '3', '4']:
            if qi not in sample_data:
                break
            
            for step, item_data in sample_data[qi]['0']:
                final_answer = item_data[qi]['0'][step]['final_answer']
                status = final_answer['selection_status']
                
                cluster_generated_answer = item_data[qi]['0'][step]['cluster_genered_answer']
                if status == "selected":
                    evidence += 1
                    continue
                else:
                    cnt += 1
                    zeroshot_answer = final_answer
                    
                    final_answer = select_final_answer(zeroshot_ans=zeroshot_answer, candidates=cluster_generated_answer)
                    if final_answer['cluster_score'] is None:
                        raw_ans = final_answer['short_answer']
                        if raw_ans in mapping_data:
                            norm_ans = mapping_data[raw_ans]
                        else:
                            norm_ans = postprocessing(raw_ans)
                            norm_ans= mapping_data.get(norm_ans, norm_ans)
                    
                        final_answer['short_answer'] = norm_ans
                        
            item_data[qi]['0'][step]['final_answer'] = final_answer 
            item_data[qi]['final_qi']['long_answer'] = final_answer['long_answer']
            
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
