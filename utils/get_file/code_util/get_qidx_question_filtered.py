import json
import csv

# Đường dẫn file
file_search = 'data/data_search_1000.json'
file_metadata = 'data/metadata_medhop.json'
file_confused = 'data/confused_samples.csv'
file_unconfused = 'data/unconfused_samples.csv'
file_out = 'data/qidx_question_9000.json'

# # Đọc metadata để lấy các qidx cần loại bỏ
# with open(file_metadata, 'r', encoding='utf-8') as f:
#     metadata = json.load(f)
# qidx_in_metadata = set(metadata.keys())


# # Đọc qidx và question từ 2 file csv, chỉ lấy qidx không nằm trong metadata
# filtered = {}
# for csv_file in [file_confused, file_unconfused]:
#     with open(csv_file, 'r', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             qidx = row.get('QIDX') or row.get('qidx') or row.get('qid')
#             question = row.get('question') or row.get('Question')
#             if qidx and question and str(qidx) not in qidx_in_metadata:
#                 filtered[str(qidx)] = question

# # Lưu ra file json
# with open(file_out, 'w', encoding='utf-8') as f:
#     json.dump(filtered, f, ensure_ascii=False, indent=2)

# print(f"Đã lưu {len(filtered)} qidx-question vào {file_out}")

with open(r'data\get_entity_9000q.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(file_search, 'r', encoding='utf-8') as f:
    cache_ner = json.load(f)

list_ner = set()
for qidx, item in data.items():
    for ent in item['entities_q']['ner']:
        ent_lower = ent.lower()
        if ent_lower not in cache_ner:
            list_ner.add(ent_lower)

list_ner = list(list_ner)
print(len(list_ner))
with open('data/entity_spacy_9000.json', 'w', encoding='utf-8') as f:
    json.dump(list_ner, f, ensure_ascii=False, indent=2)