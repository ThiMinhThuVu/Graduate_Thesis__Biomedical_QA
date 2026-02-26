import os
import json

def create_new_link_cache(folder_path, cache_file_path):
    # Khởi tạo cache trống (Tạo mới hoàn toàn)
    cache = {}
    new_entries_count = 0
    
    # Kiểm tra thư mục đầu vào
    if not os.path.exists(folder_path):
        print(f"Lỗi: Thư mục {folder_path} không tồn tại.")
        return cache

    print(f"Đang quét thư mục: {folder_path}...")

    # 1. Duyệt qua các file trong folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Trích xuất link từ cấu trúc: crawl_wiki -> html_page -> keys
                # Sử dụng .get() để tránh lỗi nếu file thiếu key
                html_page = data.get('crawl_wiki', {}).get('html_page', {})
                links = list(html_page.keys())
                
                if links:
                    # Lấy link đầu tiên tìm thấy
                    url = links[0].lower()
                    # Key là URL, Value là tên file (HashID)
                    cache[file_name] = url
                    new_entries_count += 1
            
            except Exception as e:
                print(f"Lỗi khi xử lý file {file_name}: {e}")

    # 2. Lưu file cache mới
    if new_entries_count > 0:
        # Tự động tạo thư mục chứa file cache nếu chưa có
        os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
        
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=4)
        
        print("-" * 30)
        print(f"Thành công! Đã tạo file cache mới tại: {cache_file_path}")
        print(f"Tổng số link đã lưu: {new_entries_count}")
    else:
        print("Không tìm thấy dữ liệu hợp lệ để tạo cache.")

    return cache

# --- Cấu hình đường dẫn ---
FOLDER_DATA = 'data/cache/wiki_pages_tool_1'
CACHE_FILE = os.path.join('data', 'cache', 'mapping', 'hash2url.json')

# Thực thi
new_cache = create_new_link_cache(FOLDER_DATA, CACHE_FILE)