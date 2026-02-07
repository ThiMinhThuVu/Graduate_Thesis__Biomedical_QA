import os
import json

def normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())

folder = "data/cache/crawl_wikipedia"
folder2 = "data/cache/crawl_wikipedia_tool_1_to_2"
count_empty = 0
        
path = 'data/cache/hash_id_crawl_wiki.json'
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

def is_json_empty(fpath, link):
    try:
        with open(fpath, 'r', encoding="utf-8") as f:
            content = json.load(f)
            content = content.get('crawl_wiki', {}).get('html_page', {}).get(link, None)
            if not content:
                return True
            try:
                data = content['content']
                return len(data) == 0 or data==[""]
            except Exception:
                return True
    except Exception as e:
        print(f"Lỗi đọc {fpath}: {e}")
        return True
    return False
    
# for link, filename in data.items():
#     if 'list_' in link or 'lists_' in link:
#         # count_empty += 1
#         continue
#     path1 = f"{folder}/{filename}"
#     path2 = f"{folder2}/{filename}"
#     exists = os.path.exists(path1)
#     exists2 = os.path.exists(path2)
#     empty = False
#     # if exists2 and exists:
#     #     if is_json_empty(path2, link) and is_json_empty(path1, link):
#     if exists2== True and exists == True:
#         if  is_json_empty(path2, link) == False:
#             empty = True
#             try:
#                 # os.remove(path2)
#                 # print(f"Đã xóa file rỗng: {path2}")
#                 # print(f"file 1 rỗng: {path1}")
#                 # print(f"file 2 rỗng: {path2}")
#                 print(link)
#             except Exception as e:
#                 print(f"Lỗi xóa {path2}: {e}")
#     if empty:
#         count_empty += 1
# print(f"Số mẫu content rỗng: {count_empty}")

def count_f1_diff_f2():
    """
    Đếm số file trong folder1 khác folder2 (so sánh theo tên file và nội dung).
    Trả về số file chỉ có ở folder1, số file chỉ có ở folder2, và số file có ở cả hai nhưng nội dung khác nhau.
    """
    files1 = set(os.listdir(folder))
    files2 = set(os.listdir(folder2))
    only_in_f1 = files1 - files2
    only_in_f2 = files2 - files1
    both = files1 & files2
    diff_content = 0
    for fname in both:
        f1 = os.path.join(folder, fname)
        f2 = os.path.join(folder2, fname)
        # try:
        #     with open(f1, 'rb') as file1, open(f2, 'rb') as file2:
        #         if file1.read() != file2.read():
        #             diff_content += 1
        # except Exception as e:
        #     print(f"Lỗi đọc file {fname}: {e}")
    print(f"Số file chỉ có ở folder1: {len(only_in_f1)}")
    print(f"Số file chỉ có ở folder2: {len(only_in_f2)}")
    for fname in only_in_f2:
        f1 = os.path.join(folder, fname)
        f2 = os.path.join(folder2, fname)
        # os.remove(f2)
        print('file1: ', f1)
        print('file2: ', f2)
        
    # print(f"Số file có ở cả hai nhưng nội dung khác nhau: {diff_content}")
    return len(only_in_f1), len(only_in_f2), diff_content

count_f1_diff_f2()