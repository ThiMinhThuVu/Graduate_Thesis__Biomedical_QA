import json

run = 'R11'
q_type = 'compositional'

input_path = 'data/wiki/train.json' 
pred_path = f'evaluate/run/{run}/pred_{q_type}.json'
output_file_path = f'evaluate/run/{run}/gold_{q_type}.json'

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
            'answer_id': gold_data[id]['answer_id']
        })

with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(golden, f)