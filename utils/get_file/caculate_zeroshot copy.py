import json
import  os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from postprocessing import postprocessing
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    

dict_golden = load_json(r'data\all_answer_golden_56_update_v0.json')
dict_o3 = load_json(r'data\cache\O3_zeroshot_ans_cache.json')
mapping_data = load_json(r'data/mapping_answer_gol_add_seq.json')
cnt = 0
cnt_true = 0
# BASE = "data/ALL_ROUND/medhop_results_ct6_r36_parallel_seq/parallel_decomp_new_ct6/all_ans"

meta = load_json(r'data\metadata_medhop.json')
BASE = "ZEROSHOT_SEQ_PAR_medhop_results/all_ans"
for ID, item in meta.items():
    if item['type_q'] != 'parallel' or item['target_answer_type'] == 'correctness' or dict_golden[ID]['True'] == []:
        continue
    file = f"{BASE}/R4_step_3b_all_ans_{ID}.json"
    sample = load_json(file)
    
    cnt += 1
    ans = sample['final_answer']['short_answer'].lower()
    if ans in dict_golden[ID]['True']:
        cnt_true += 1

print(cnt, cnt_true)

            
# for ID in miss_id:
#     zero_ans = dict_o3[ID]['short_answer'].lower()
    
#     dict_candidate = wh[ID]
#     for label, cluster in dict_candidate.items():
#         ans = cluster['llm']['short_answer'].lower()
#         if ans != 'insufficient information' and postprocessing(ans) == postprocessing(zero_ans):
#             cnt += 1
#             print(ID, zero_ans, ans)
#             if ans in dict_golden[ID]['True']:
#                 cnt_true += 1
#             break
        # if ans != 'insufficient information':
        #     if zero_ans in mapping_data:
        #         norm_zero = mapping_data[zero_ans]
        #     else:
        #         norm_zero = mapping_data.get(postprocessing(zero_ans), postprocessing(zero_ans))
              
        #     if ans in mapping_data:
        #         norm_ans = mapping_data[ans]
        #     else:
        #         norm_ans = mapping_data.get(postprocessing(ans), postprocessing(ans))
            
        #     if norm_zero == norm_ans:                  
        #         cnt += 1
        #         print(ID, zero_ans, ans)
        #         if norm_ans in dict_golden[ID]['True']:
        #             cnt_true += 1
        #         break
# print(cnt_true, cnt, cnt_true/cnt)
