import sys
import json
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tqdm import tqdm


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
dict_golden = load_json(r'data\all_answer_golden_56.json')
golden_mapping = load_json(r'mapping_golden.json')
# input_data = load_json(r'src\cluster\cache\answer_k_mean_300_3.json')
source = 'wiki' #"wiki' #'google'
s_type = 'kmean' #'kmean' #'hdbscan' #check label
num_cluster = '7' # "_3"

# input_data = load_json(fr'src\cluster\final\concat_answer_{s_type}_merge_google_wiki_{num_cluster}.json')
input_data = load_json(r"src\cluster\final\YN_answer_kmean_merge_google_wiki_7.json")
cnt_label = [0] * 15
cnt_cluster = [0]*15
for qidx, item in input_data.items():
    t = 0
    list_golden = dict_golden[qidx]['True']
    list_alias = []
    
    for gol in list_golden:
        gol = gol.strip()
        list_alias += golden_mapping[gol]
    
    cnt_cluster[len(item.keys())] += 1
    for label, cluster_sample in item.items():
        # if label == '-1':
        #     continue
        for alias in list_alias:
            if alias.lower() in cluster_sample['context'].lower():
                t += 1
                break
    cnt_label[t] += 1
    # if t == 0:
    #     print(qidx)
print(cnt_label)
print(cnt_cluster)

