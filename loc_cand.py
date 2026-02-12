import json
import os

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

metadata = load_json("data/metadata_medhop.json")

folder_300 = "medhop_results_no_mcq_TFIDF300/all_ans"
folder_300_mcq = "medhop_results_mcq_2_TFIDF1000/all_ans"

cnt = 0
for qidx, item in metadata.items():
    filepath1 = f"{folder_300}/R4_step_3b_all_ans_{qidx}.json"
    filepath2 = f"{folder_300_mcq}/R4_step_3b_all_ans_{qidx}.json"

    if os.path.exists(filepath1) and os.path.exists(filepath2):
        data1 = load_json(filepath1)
        data2 = load_json(filepath2)
        # if data2['target_answer_type'] != 'correctness':
        #     continue
        
        source_output1 = data1['1']['0']['0']['source_output']
        source_output2 = data2['1']['0']['0']['source_output']
        
        for src, item in source_output1.items():
            if src in ['google', 'wiki', 'crawl_sparql_wikidata']:
                data2['1']['0']['0']['source_output'][src] = item
            
        # ans1 = source_output1['wiki']['candidate'][0]['short_answer'].lower()
        # ans2 = source_output2['wiki']['candidate'][0]['short_answer'].lower()
        
        # ans1 = source_output1['crawl_sparql_wikidata']['short_answer'].lower()
        # ans2 = source_output2['crawl_sparql_wikidata']['short_answer'].lower()
        
        # if ans1 != ans2:
        #     cnt += 1
        #     print("FILE 1: ", filepath1)
        #     print("FILE 2: ", filepath2)
        #     print("======================")
        # for src in list(source_output.keys()):
            # if src not in ['crawl_wiki', 'crawl_ner_wiki']:
            #     del source_output[src]
            
        
        save_json(data2, filepath2)
        print(filepath2)
        
print(cnt)