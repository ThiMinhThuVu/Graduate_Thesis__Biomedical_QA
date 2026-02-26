import json
import os

def load_json(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except:
            return {}

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- PHẦN 1: MERGE VÀ CHUẨN HÓA url2hash.json ---
url2hash_main_path = r'2wiki\cache\mapping\url2hash.json'
url_data = load_json(url2hash_main_path)
print(f"Số lượng url2hash gốc: {len(url_data)}")

# Giả sử các file lẻ nằm trong thư mục data\cache\mapping\url\
# Gộp từ url2hash0.json đến url2hash20.json (bạn thay số 21 bằng i+1)
for i in range(6): 
    filepath = fr'2wiki\cache\mapping\entity\url2hash{i}.json'
    if os.path.exists(filepath):
        sub_data = load_json(filepath)
        url_data.update(sub_data)
        print(f"Đã gộp file URL: {filepath} (+{len(sub_data)} records)")

# Chuẩn hóa toàn bộ key về lowercase sau khi gộp
final_url2hash = {k.lower().strip(): v for k, v in url_data.items()}

save_json(final_url2hash, url2hash_main_path)
print(f"Tổng số url2hash sau khi gộp & chuẩn hóa: {len(final_url2hash)}")


# --- PHẦN 2: MERGE entity2wiki.json ---
entity_main_path = r'2wiki\cache\mapping\entity2wiki.json'
entity_data = load_json(entity_main_path)
print(f"\nSố lượng entity gốc: {len(entity_data)}")

for i in range(6):
    filepath = fr'2wiki\cache\mapping\entity\entity2wiki{i}.json'
    if os.path.exists(filepath):
        sub_data = load_json(filepath)
        entity_data.update(sub_data)
        print(f"Đã gộp file Entity: {filepath} (+{len(sub_data)} records)")

save_json(entity_data, entity_main_path)
print(f"Tổng số entity sau khi gộp: {len(entity_data)}")