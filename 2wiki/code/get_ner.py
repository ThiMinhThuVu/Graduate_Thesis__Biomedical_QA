import json
import os

def load_json(path):
    if not os.path.exists(path):
        print(f"Lỗi: Không tìm thấy file tại {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file {path}: {e}")
        return {}

def save_json(data, file_path):
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        # Chuyển set thành list vì JSON không hỗ trợ set
        json.dump(list(data), file, ensure_ascii=False, indent=4)
    print(f"Đã lưu file tại: {file_path}")
input_ner = load_json("2wiki/all_ner_test.json")

list_ner = set()

for ner in input_ner:
    list_ner.add(ner.lower())
# list_folder = [
#     "D:/DSKTLab/Test_2wiki/comparison/wiki_preprocessing/step1",
#     "D:/DSKTLab/Test_2wiki/bridge_comparison/wiki_preprocessing/step1",
#     "D:/DSKTLab/Test_2wiki/compositional/wiki_preprocessing/step1",
#     "D:/DSKTLab/Test_2wiki/inference/wiki_preprocessing/step1"
# ]

# for base_folder in list_folder:
#     if not os.path.exists(base_folder):
#         print(f"Bỏ qua: Thư mục không tồn tại -> {base_folder}")
#         continue
        
#     # Duyệt qua tất cả các thư mục con và file
#     for root, dirs, files in os.walk(base_folder):
#         for file_name in files:
#             if file_name.endswith(".json"):
#                 file_path = os.path.join(root, file_name)
#                 data = load_json(file_path)
                
#                 # Duyệt qua cấu trúc dictionary của bạn
#                 for path_i, path_item in data.items():
                 
#                     for step_i, step_item in path_item.items():
#                         if isinstance(step_item, dict) and step_item.get('key_entity'):
#                             list_ner.add(step_item['key_entity'])

# Lưu kết quả
save_json(list_ner, "2wiki/all_ner_test.json")
print(len(list_ner))