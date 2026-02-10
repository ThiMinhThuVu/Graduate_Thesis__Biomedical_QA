import json

# Đường dẫn các file nguồn
file1 = 'data/meta_entity_v2.json'
file2 = 'data/meta_entity_YN.json'
output_file = 'data/metadata.json'

# Đọc file thứ nhất
with open(file1, 'r', encoding='utf-8') as f:
    data1 = json.load(f)

# Đọc file thứ hai
with open(file2, 'r', encoding='utf-8') as f:
    data2 = json.load(f)

# Gộp hai dict lại
data = {**data1, **data2}

# Xóa trường "true ans" và "false ans" nếu có
for v in data.values():
    v.pop("true ans", None)
    v.pop("false ans", None)

# Ghi lại file JSON đã chỉnh sửa
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)