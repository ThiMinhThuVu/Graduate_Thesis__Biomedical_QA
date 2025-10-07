import os
import json
from collections import Counter
import matplotlib.pyplot as plt

folder_path = r"E:\UETQuintet\wiki_inference_results\all_ans"

def count_qid_per_file(folder_path):
    histogram = Counter()

    # Lặp qua tất cả các file JSON trong thư mục
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file).get('final_answer', {})
                    
                    # Kiểm tra có trường normalized_v2 -> parent không
                    parent_data = data.get("normalized_v2", {}).get("parent")
                    if parent_data and isinstance(parent_data, dict):
                        # Lọc các QID có giá trị không rỗng
                        valid_parents = []
                        for qid, values in parent_data.items():
                            if len(values) > 0:
                                valid_parents.append(qid)
                                
                        num_qid = len(valid_parents)
                        if num_qid == 2:
                            print(f"Parent data in file {file_path}")
                        if num_qid > 0:
                            histogram[num_qid] += 1  # đếm theo số lượng QID tìm được
            except json.JSONDecodeError:
                print(f"Không thể đọc file JSON: {file_name}")
            except Exception as e:
                print(f"Lỗi khi xử lý {file_name}: {e}")

    return histogram

# Gọi hàm
histogram = count_qid_per_file(folder_path)

# In ra thống kê
print("\nSố lượng file có parent theo số QID tìm được:")
for num_qid, count in sorted(histogram.items()):
    print(f"{num_qid} QID: {count} files")

# Vẽ histogram
plt.bar(histogram.keys(), histogram.values())
plt.xlabel("Số lượng QID tìm được trong parent")
plt.ylabel("Số lượng file JSON")
plt.title("Phân bố số lượng QID tìm được trong parent (per file)")
plt.xticks(sorted(histogram.keys()))
plt.show()
