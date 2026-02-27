import json

def parse_batch_output(input_jsonl_path, output_json_path):
    final_results = {}
    
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            # 1. Parse dòng JSONL
            data = json.loads(line)
            
            custom_id = data.get("custom_id")
            response = data.get("response")
            
            # Kiểm tra xem request có thành công không (status_code 200)
            if response and response.get("status_code") == 200:
                # 2. Lấy nội dung message content (đang ở dạng string JSON)
                content_str = response["body"]["choices"][0]["message"]["content"]
                
                try:
                    # 3. Parse chuỗi content thành dict thực sự
                    content_data = json.loads(content_str)
                    
                    # Lưu vào dict kết quả với key là custom_id
                    final_results[custom_id] = {
                        "QIDX": custom_id,
                        "short_answer": content_data.get("short_answer", []),
                        "long_answer": content_data.get("long_answer", []),
                        # Bạn có thể lưu thêm usage nếu cần quản lý chi phí
                        "usage": response["body"].get("usage", {})
                    }
                except json.JSONDecodeError:
                    print(f"❌ Lỗi parse JSON nội dung tại ID: {custom_id}")
            else:
                error_info = data.get("error")
                print(f"⚠️ Request ID {custom_id} thất bại: {error_info}")

    # 4. Lưu kết quả cuối cùng ra file .json
    with open(output_json_path, 'w', encoding='utf-8') as out_f:
        json.dump(final_results, out_f, ensure_ascii=False, indent=4)
        
    print(f"✅ Đã parse xong! Lưu {len(final_results)} records vào: {output_json_path}")

# Sử dụng
if __name__ == "__main__":
    # Thay đường dẫn file thực tế của bạn vào đây
    input_path = r"src\batch_api\output\zeroshot_o3_v2_3500_9000.jsonl" 
    output_path = r"src\batch_api\output\format\zeroshot_o3_v2_3500_9000.json"
    
    parse_batch_output(input_path, output_path)