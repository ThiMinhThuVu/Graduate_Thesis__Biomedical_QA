import os
import json

def delete_retrieval():
    # Đường dẫn tới thư mục chứa các file json
    folder = r"e:/MAS_UETQuintet/data/cache/retrieval/"

    count = 0
    for filename in os.listdir(folder):
        if filename.endswith('.json'):
            filepath = os.path.join(folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'google' in data and isinstance(data['google'], dict) and data['google'].get('search_results', None) in [None, []]:
                    os.remove(filepath)
                    count += 1
                    print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Error with {filename}: {e}")
    print(f"Total deleted: {count}")

def delete_ans_wrong():
    # Đường dẫn tới thư mục chứa các file json
    folder = r"medhop_results/all_ans"
    medhop_root = r"medhop_results"
    deleted_ids = set()
    count = 0
    # Xóa file trong all_ans và lấy id
    for filename in os.listdir(folder):
        if filename.endswith('.json'):
            filepath = os.path.join(folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if (
                    'final_answer' in data
                    and isinstance(data['final_answer'], dict)
                    and (data['final_answer'].get('retrieval_information', None) in ["\n\n\n\n", "", []]
                         or len(data['1']['final_qi']['retrieval_information']) <= 10
                         or data['1']['final_qi']['retrieval_information'] in ["\n\n\n\n", "", []])
                ):
                    # Lấy id từ tên file (bỏ đuôi .json)
                    file_id = os.path.splitext(filename)[0].split('_')[-1]
                    deleted_ids.add(file_id)
                    os.remove(filepath)
                    count += 1
                    print(f"Deleted in all_ans: {filename}")
            except Exception as e:
                print(f"Error with {filename}: {e}")

    # Xóa file trong các folder con của medhop_results nếu tên file chứa id đã xóa
    for subfolder in os.listdir(medhop_root):
        subfolder_path = os.path.join(medhop_root, subfolder)
        if os.path.isdir(subfolder_path) and subfolder != 'all_ans':
            for filename in os.listdir(subfolder_path):
                for file_id in deleted_ids:
                    if file_id in filename and filename.endswith('.json'):
                        try:
                            filepath = os.path.join(subfolder_path, filename)
                            os.remove(filepath)
                            print(f"Deleted in {subfolder}: {filename}")
                        except Exception as e:
                            print(f"Error deleting {filename} in {subfolder}: {e}")
                            
    print(f"Total deleted in all_ans: {count}")

delete_retrieval()