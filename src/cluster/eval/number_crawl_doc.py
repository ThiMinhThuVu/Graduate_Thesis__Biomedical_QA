import json


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_not_selected_histogram(original_path, cluster_path, max_docs=100):
    data_original = load_json(original_path)
    data_cluster = load_json(cluster_path)

    # 🔥 Mảng đếm
    count_not_selected = [0] * 18

    for qidx, docs in data_original.items():

        # ===== 1️⃣ Lấy hash_id gốc =====
        original_hash_ids = set()
        for d in docs:
            if "hash_id" in d:
                original_hash_ids.add(d["hash_id"])

        N_total = len(original_hash_ids)

        # ===== 2️⃣ Lấy hash_id được chọn =====
        selected_hash_ids = set()

        if qidx in data_cluster:
            clusters = data_cluster[qidx]

            for label_c, cluster in clusters.items():
                # if label_c == '-1':
                #     continue
                
                if "hash_id" in cluster:
                    for hid in cluster["hash_id"]:
                        selected_hash_ids.add(hid)

        # 🔥 Chỉ giữ doc nằm trong original
        selected_hash_ids = selected_hash_ids & original_hash_ids

        N_selected = len(selected_hash_ids)
        N_not_selected = N_total - N_selected

        # ===== 3️⃣ Update mảng đếm =====
        if 0 <= N_not_selected <= max_docs:
            count_not_selected[N_not_selected] += 1
        else:
            print(f"Warning: sample {qidx} vượt max_docs")

    return count_not_selected


if __name__ == "__main__":
    s_type = 'kmean' #check label_c == '-1'
    n_cluster = '7'
    original_file = r"data\cache\cache_crawl_data_doc.json"
    # cluster_file = fr"src\cluster\final\output_cluster_{s_type}_euclidean_top5_{n_cluster}.json"
    cluster_file = r"src\cluster\final\YN_answer_kmean_merge_google_wiki_7.json"
    print(cluster_file)
    histogram = compute_not_selected_histogram(original_file, cluster_file, max_docs=50)

    print("Histogram NOT selected:")
    print(histogram)
