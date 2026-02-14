import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from postprocessing import postprocessing

def load_golden(golden_path):
    with open(golden_path, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)
    return golden_data

dict_golden = load_golden(r"data\all_answer_golden_56.json")
meta_mapping = load_golden(r"data\mapping_answer_gol.json")

dict_output = load_golden(r"batch_output_58_results_split.json")
# dict_output = load_golden(r"o3_mini_mcq_wh_58.json")


cnt = 0

for qidx, item in dict_output.items():
    list_gol = dict_golden[qidx]['True']
    if list_gol == []:
        continue
    
    ans = item['short_answer']
    if ans in meta_mapping:
        ans = meta_mapping[ans]
    else:
        ans = postprocessing(ans)
        ans = meta_mapping.get(ans, ans)
    
    item['golden'] = list_gol
    if ans in list_gol:
        cnt += 1
        item['check'] = True
    else:
        item['check'] = False
print(cnt, '/58')

with open('batch_output_58_results_split.json', 'w', encoding='utf-8') as f:
    json.dump(dict_output, f, indent=2,ensure_ascii=False)



# for qidx, item in dict_output.items():
#     list_gol = dict_golden[qidx]['True']
#     if list_gol == []:
#         continue
    
#     ans = item['mcq']['short_answer']
#     if ans in meta_mapping:
#         ans = meta_mapping[ans]
#     else:
#         ans = postprocessing(ans)
#         ans = meta_mapping.get(ans, ans)
    
#     item['golden'] = list_gol
#     if ans in list_gol:
#         cnt += 1
#         item['check'] = True
#     else:
#         item['check'] = False
# print(cnt, '/58')

# with open('o3_mini_mcq_wh_58.json', 'w', encoding='utf-8') as f:
#     json.dump(dict_output, f, indent=2,ensure_ascii=False)