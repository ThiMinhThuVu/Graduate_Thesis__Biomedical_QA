import json
import re


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def add_reference_count(cluster_path, output_path=None):

    data = load_json(cluster_path)

    for qidx, clusters in data.items():

        for cid, cluster in clusters.items():

            ref_cnt = 0

            if "llm" in cluster and "long_answer" in cluster["llm"]:

                long_answer = cluster["llm"]["long_answer"]

                # 🔥 Tìm tất cả pattern dạng [....]
                refs = re.findall(r"\[[^\]]+\]", long_answer)

                ref_cnt = len(refs)

            # ===== thêm field mới =====
            cluster["ref_cnt"] = ref_cnt

    # Nếu không truyền output_path → ghi đè file cũ
    if output_path is None:
        output_path = cluster_path

    save_json(data, output_path)

    print("Done. Added ref_cnt to each cluster.")
    return data


if __name__ == "__main__":
    s_type = 'kmean'
    n_cluster = 7
    cluster_file = fr"src\cluster\final\concat_answer_{s_type}_merge_google_wiki_7.json"
    output_path = fr"src\cluster\final\ref_cnt.json"
    data = add_reference_count(cluster_file, output_path)
