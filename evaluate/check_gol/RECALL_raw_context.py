import sys
import json
import os


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

metadata = load_json(r'all_inf_direct_300.json')
golden = load_json(r'data\all_answer_golden_56.json')
raw_data = load_json(r'data\metadata_medhop.json')

list_id = [
    256957,
    284028,
    290551,
    419700,
    480829,
    651642,
    722978,
    879116,
    888878,
    970294
]

output = {}

for qidx in list_id:
    qidx = str(qidx)
    list_golden = golden[qidx]['True']
    context = metadata[qidx].lower()
    output[qidx] = {
        'check': 0,
        'target_answer': raw_data[qidx]['target_answer_type']
    }
    for gol in list_golden:
        if gol in context:
            output[qidx]['check'] = 1
            break
    
print(output)
    
