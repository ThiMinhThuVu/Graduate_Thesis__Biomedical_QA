import json
import os
import sys
from tqdm import tqdm 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.hash import make_cache_filename
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_and_aggregate_docs(file_paths, current_cache):
    """
    Đọc tất cả các file input và gom các doc chưa có trong cache.
    """
    new_docs_to_encode = {} # Chỉ chứa những hash_id cần phải tính embedding
    
    for path in file_paths:
        if not os.path.exists(path):
            print(f"⚠️ Cảnh báo: Không tìm thấy file {path}")
            continue
            
        print(f"--- Đang đọc file: {path} ---")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for qid, content in tqdm(data.items(), desc=f"Trích xuất {os.path.basename(path)}"):
            contexts = content.get("context", [])
            for item in contexts:
                title = str(item[0]).strip().lower()
                sentences = item[1]
                
                hash_id = make_cache_filename(title)
                full_doc = " ".join(sentences)
                
                # Điều kiện: Nếu chưa có trong cache gốc VÀ chưa nằm trong danh sách chờ encode
                if hash_id not in current_cache and hash_id not in new_docs_to_encode:
                    new_docs_to_encode[hash_id] = {
                        "doc": full_doc,
                        "embedding": None
                    }
    return new_docs_to_encode


# --- Cấu hình đường dẫn ---
input_files = [
    r'data\wiki\test\test_compositional.json',
    r'data\wiki\test\test_comparison.json',
    r'data\wiki\test\test_bridge.json',
    r'data\wiki\test\test_inference.json' # Thay đổi tên file cho đúng thực tế của bạn
]
output_path = '2wiki/cache/embedding/embedding_cache_ctx_original.json'

os.makedirs(os.path.dirname(output_path), exist_ok=True)
if os.path.exists(output_path):
    print("--- Đang load cache cũ từ đĩa... ---")
    with open(output_path, 'r', encoding='utf-8') as f:
        cache_embedding = json.load(f)
else:
    cache_embedding = {}

pending_docs = load_and_aggregate_docs(input_files, cache_embedding)

if pending_docs:
    hash_ids = list(pending_docs.keys())
    docs_to_encode = [pending_docs[hid]["doc"] for hid in hash_ids]
    
    print(f"Có {len(docs_to_encode)} title mới. Bắt đầu tạo embedding...")
    embeddings = embedding_model.encode(docs_to_encode, batch_size=32, show_progress_bar=True)
    
    # Gán embedding vào pending_docs
    for i, hid in enumerate(tqdm(hash_ids, desc="Cập nhật vector")):
        pending_docs[hid]["embedding"] = embeddings[i].tolist()
    
    cache_embedding.update(pending_docs)
    
    print(f"--- Đang lưu cache đã cập nhật ({len(cache_embedding)} records) ---")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cache_embedding, f, ensure_ascii=False, indent=4)
else:
    print("Không có dữ liệu mới cần tạo embedding. Cache đã đầy đủ.")

print("--- Hoàn thành! ---")