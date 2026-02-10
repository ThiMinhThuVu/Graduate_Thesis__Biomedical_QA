import os
import json

# Đường dẫn folder chứa các file json
json_folder = r"medhop_results_mcq_3_TFIDF300/all_ans"

for fname in os.listdir(json_folder):
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(json_folder, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Lấy dict source_output
    try:
        source_output = data["1"]["0"]["0"]["source_output"]
    except Exception as e:
        print(f"[WARN] {fname}: {e}")
        continue

    # Tạo list candidate mới từ các source
    new_candidates = []
    for src, result in source_output.items():
        short_answer = result.get("short_answer", "")
        long_answer = result.get("long_answer", "")
        retrieval_information = result.get("retrieval_information", "")
        new_candidates.append({
            "source": src,
            "short_answer": short_answer,
            "long_answer": long_answer,
            "retrieval_information": retrieval_information
        })

    # Gán vào final_qi['candidate']
    if "1" in data and "final_qi" in data["1"]:
        data["1"]["final_qi"]["candidate"] = new_candidates
        data["1"]["final_qi"]["short_answer"] = ""
        data["1"]["final_qi"]["long_answer"] = ""
        data["1"]["final_qi"]["retrieval_information"] = ""
    
        if "final_answer" in data:
            del data["final_answer"]
        
            # Ghi đè lại file
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[OK] Updated: {fname}")
    else:
        print(f"[WARN] {fname}: Không tìm thấy ['1']['final_qi']!")
