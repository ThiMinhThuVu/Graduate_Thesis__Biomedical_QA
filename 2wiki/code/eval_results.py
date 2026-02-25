import json
import  os

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

metadata = load_json(r'2wiki\meta_demo_comparison_200q.json')
BASE = "E:/MAS_UETQuintet/2wiki/cache/ALL_ROUND/DEMO_COMPARISON_wiki_results/all_ans"

cnt = 0
no_evi = 0
output = {}
cnt_true = 0
for qidx, item in metadata.items():
    qidx = item['QIDX']
    file_path = f"{BASE}/R4_step_3b_all_ans_{qidx}.json"
    golden = item['answer'].lower()
    if os.path.exists(file_path):
        
        
        cnt += 1
        data = load_json(file_path)
        # final_answer = data['1']['0']['0']['final_answer']
        short_ans =  data['final_answer']['short_answer'].lower()
        
        
        if short_ans == 'hello thu':
            no_evi += 1
            
        if short_ans == golden:
            cnt_true += 1
            check = True
        else:
            check = False
            print("golden: ", golden)
            print("short_ans: ", short_ans)
            print(file_path)
        output[qidx] = {
            'short_ans': short_ans,
            'golden': golden,
            'check': check
        }
        
save_json(output, r'2wiki\cache\eval\comparison_100q.json')
print(cnt_true, no_evi, cnt)