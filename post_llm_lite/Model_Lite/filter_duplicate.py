import json

def filter_duplicate_custom_ids(input_path, output_path):
    seen_ids = set()
    unique_data = []
    
    # Đọc tất cả các dòng từ file gốc
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_before = len(lines)
    
    # Lọc trùng
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        try:
            item = json.loads(line)
            c_id = item.get("custom_id")
            
            # Chỉ giữ lại nếu custom_id chưa từng xuất hiện
            if c_id not in seen_ids:
                seen_ids.add(c_id)
                unique_data.append(item)
        except json.JSONDecodeError:
            print(f"⚠️ Bỏ qua dòng lỗi định dạng JSON")

    total_after = len(unique_data)

    # Ghi file mới
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in unique_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("-" * 30)
    print(f"📊 Kết quả xử lý:")
    print(f"🔹 Độ dài trước khi lọc: {total_before} dòng")
    print(f"🔹 Độ dài sau khi lọc:  {total_after} dòng")
    print(f"✂️ Đã loại bỏ:         {total_before - total_after} dòng trùng lặp")
    print(f"✅ File sạch đã lưu tại: {output_path}")
    print("-" * 30)

# Cấu hình đường dẫn file của bạn
input_file = "post_llm_lite/submission_lite_model_10000q_merged.jsonl"
output_file = "post_llm_lite/Model_Lite/submission_lite_model_10000q_fix_duplicate.jsonl"

filter_duplicate_custom_ids(input_file, output_file)