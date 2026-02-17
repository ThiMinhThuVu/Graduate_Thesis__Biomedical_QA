from collections import defaultdict
import sys
import json
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from postprocessing import postprocessing


# =====================
# UTILS
# =====================
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize(text):
    return text.strip().lower()


# =====================
# LOAD DATA
# =====================
golden = load_json(r'data\all_answer_golden_56.json')
metadata = load_json(r'src\cluster\cache\answer_k_mean_300_3.json')
metadata_2 = load_json(r'src\cluster\cache\answer_k_mean_300_5.json')

mapping_data = load_json(r'data\mapping_answer_gol.json')


# =====================
# METRICS
# =====================
total_questions = 0

question_correct = 0            # >=1 cluster đúng
question_none_correct = 0       # không cluster nào đúng
all_no_evidence_questions = 0   # tất cả cluster đều no-evidence

cluster_correct = defaultdict(int)
cluster_total = defaultdict(int)
cluster_no_evidence = defaultdict(int)

total_no_evidence = 0

def merge_result(metadata_1, metadata_2):
    all_qidx = set(metadata_1.keys()) | set(metadata_2.keys())

    for qidx in all_qidx:

        if qidx not in golden:
            continue

        total_questions += 1
        list_golden = [normalize(x) for x in golden[qidx]["True"]]

        hit = False
        all_no_evidence = True

        # lấy clusters từ cả 2 metadata
        clusters_1 = metadata_1.get(qidx, {})
        clusters_2 = metadata_2.get(qidx, {})

        # gộp lại thành 1 dict (cluster id sẽ có prefix để tránh trùng)
        merged_clusters = {}

        for cid, data in clusters_1.items():
            merged_clusters[f"m1_{cid}"] = data

        for cid, data in clusters_2.items():
            merged_clusters[f"m2_{cid}"] = data

        # =====================
        # LOOP CLUSTERS
        # =====================
        for cluster_id, cluster_data in merged_clusters.items():

            if "llm" not in cluster_data:
                continue

            pred = normalize(cluster_data["llm"].get("short_answer", ""))

            # ---------------------
            # NO EVIDENCE
            # ---------------------
            if pred == "insufficient information":
                cluster_no_evidence[cluster_id] += 1
                total_no_evidence += 1
                continue
            else:
                all_no_evidence = False

            # ---------------------
            # POSTPROCESS + MAP
            # ---------------------
            if pred in mapping_data:
                pred = mapping_data[pred]
            else:
                pred = postprocessing(pred)
                pred = mapping_data.get(pred, pred)

            cluster_total[cluster_id] += 1

            # ---------------------
            # CHECK CORRECT
            # ---------------------
            if pred in list_golden:
                cluster_correct[cluster_id] += 1
                hit = True

        # ---------------------
        # QUESTION-LEVEL
        # ---------------------
        if hit:
            question_correct += 1
        else:
            question_none_correct += 1

        if all_no_evidence:
            all_no_evidence_questions += 1



# =====================
# MAIN LOOP
# =====================
for qidx, clusters in metadata.items():

    if qidx not in golden:
        continue

    total_questions += 1
    list_golden = [normalize(x) for x in golden[qidx]["True"]]

    hit = False
    all_no_evidence = True

    for cluster_id, cluster_data in clusters.items():

        if "llm" not in cluster_data:
            continue

        pred = normalize(cluster_data["llm"].get("short_answer", ""))

        # ---------------------
        # NO EVIDENCE
        # ---------------------
        if pred == "insufficient information":
            cluster_no_evidence[cluster_id] += 1
            total_no_evidence += 1
            continue
        else:
            all_no_evidence = False

        # ---------------------
        # POSTPROCESS + MAP
        # ---------------------
        if pred in mapping_data:
            pred = mapping_data[pred]
        else:
            pred = postprocessing(pred)
            pred = mapping_data.get(pred, pred)

        cluster_total[cluster_id] += 1

        # ---------------------
        # CHECK CORRECT
        # ---------------------
        if pred in list_golden:
            cluster_correct[cluster_id] += 1
            hit = True

    # ---------------------
    # QUESTION-LEVEL
    # ---------------------
    if hit:
        question_correct += 1
    else:
        question_none_correct += 1

    if all_no_evidence:
        all_no_evidence_questions += 1


# =====================
# MAIN LOOP (MERGE 2 METADATA)
# =====================



# =====================
# PRINT RESULTS
# =====================
print("=" * 50)
# print("TOTAL QUESTIONS:", total_questions)

print("\nQUESTION-LEVEL METRICS")
has_evidence = total_questions - all_no_evidence_questions
print(f"Has evidence        : {has_evidence}/{total_questions}")
print(f"All no-evidence     : {all_no_evidence_questions}/{total_questions} "
      f"({round(all_no_evidence_questions/total_questions,4)})")

print(f">=1 correct cluster : {question_correct}/{has_evidence} "
      f"({round(question_correct/has_evidence,4)})")

print(f"No correct cluster  : {question_none_correct}/{has_evidence} "
      f"({round(question_none_correct/has_evidence,4)})")

# print("\nCLUSTER-LEVEL ACCURACY (exclude no-evidence)")
# for cid in sorted(cluster_total.keys()):
#     acc = cluster_correct[cid] / cluster_total[cid] if cluster_total[cid] > 0 else 0
#     print(f"Cluster {cid}: {acc:.4f} "
#           f"({cluster_correct[cid]}/{cluster_total[cid]})")

# print("\nNO-EVIDENCE STATISTICS")
# for cid in sorted(cluster_no_evidence.keys()):
#     print(f"Cluster {cid}: {cluster_no_evidence[cid]} no-evidence")

# print("\nTotal no-evidence predictions:", total_no_evidence)
# print("=" * 50)
