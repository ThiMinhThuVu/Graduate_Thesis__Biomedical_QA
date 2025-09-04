import json

type_q = "inference"
input_path = 'data/wiki/dev.json' 
pred_path = f'evaluate/run/evidence/pred_{type_q}.json'
output_file_path = f'evaluate/run/evidence/gold_{type_q}.json'

gold_data = {} 
with open(input_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

for item in data:
    question_id = item['_id']
    new_item = {key: value for key, value in item.items() if key != '_id'}
    gold_data[question_id] = new_item

with open(pred_path, 'r', encoding='utf-8') as file:
    pred_data = json.load(file)

golden = []
for id in pred_data['answer']: 
    if id in gold_data:
        golden.append({
            "_id": id,
            "answer": gold_data[id]['answer'],
            'answer_id': gold_data[id]['answer_id'],
            'supporting_facts': gold_data[id]['supporting_facts'],
            'evidences': gold_data[id]['evidences'],
            'evidences_id': gold_data[id]['evidences_id']
        })

with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(golden, f)