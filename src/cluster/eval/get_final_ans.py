# =========================
# IMPORTS
# =========================
import json
import re
import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from postprocessing import postprocessing

# =========================
# CONFIG
# =========================

DISTANCE_THRESHOLD = 0.12

USE_SOURCE_BONUS = False # False # True
SOURCE_DISTANCE_MARGIN = 0.05

GG_BONUS = 0.2
WIKI_BONUS = 0.3


# =========================
# JSON UTILS
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

mapping_data = load_json(r'data\mapping_answer_gol.json')


# =========================
# REFERENCE PARSING
# =========================
def extract_references(long_answer):
    if not long_answer:
        return []
    return re.findall(r"\[([^\]]+)\]", long_answer)


def extract_source(ref):
    parts = ref.split("_")
    source_parts = []
    for p in parts:
        if p.isdigit():
            break
        source_parts.append(p)
    return "_".join(source_parts)


def detect_source_from_long_answer(long_answer):

    refs = extract_references(long_answer)
    sources = [extract_source(r) for r in refs]

    has_wiki = any("WIKI" in s for s in sources)
    has_gg = any("GG" in s for s in sources)

    return has_wiki, has_gg, len(refs)


# =========================
# SIMPLE COHESION
# =========================
def cluster_cohesion(embeddings):
    if len(embeddings) <= 1:
        return 0
    centroid = embeddings.mean(axis=0)
    dists = np.linalg.norm(embeddings - centroid, axis=1)
    return dists.mean()


# =========================
# MAIN CLUSTER EVALUATION
# =========================
def cluster_and_evaluate(cluster_file, model):

    data = load_json(cluster_file)
    fused_results = {}

    for qidx, clusters in data.items():

        cand_list = []

        for cid, cluster in clusters.items():

            if "llm" not in cluster:
                continue

            ans = cluster["llm"].get("short_answer", "")
            ans = ans.strip().lower()

            long_answer = cluster["llm"].get("long_answer", "")
            context = cluster.get("context", "").lower()
            ref_cnt = cluster.get('ref_cnt', 0)
            
            # ===== FILTER NO-EVIDENCE =====
            if not ans or ans == "insufficient information":
                continue

            # ===== NORMALIZE =====
            if ans in mapping_data:
                norm_ans = mapping_data[ans]
            else:
                norm_ans = postprocessing(ans)
                norm_ans = mapping_data.get(norm_ans, norm_ans)

            refs = extract_references(long_answer)
            sources = [extract_source(r) for r in refs]

            has_wiki = any("WIKI" in s for s in sources)
            has_gg = any("GG" in s for s in sources)

            cand_list.append({
                "text": norm_ans,
                "long_answer": long_answer,
                "context": context,
                "sources": sources,          # <-- thêm dòng này
                "has_wiki": has_wiki,
                "has_gg": has_gg
            })

        if len(cand_list) < 2:
            fused_results[qidx] = {"abstain": True}
            continue

        texts = [c["text"] for c in cand_list]
        embeddings = model.encode(texts)

        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=DISTANCE_THRESHOLD,
            metric="cosine",
            linkage="average"
        ).fit(embeddings).labels_

        best_cluster = None
        best_cohesion = 1e9
        raw_clusters = []
        cluster_sources = set()

        for lbl in set(labels):

            indices = [i for i, l in enumerate(labels) if l == lbl]
            evidence_count = len(indices)

            has_wiki = False
            has_gg = False

            for i in indices:
                wiki_flag, gg_flag, _ = detect_source_from_long_answer(
                    cand_list[i]["long_answer"]
                )
                if wiki_flag:
                    has_wiki = True
                if gg_flag:
                    has_gg = True
                    
                cluster_sources.update(cand_list[i]["sources"])

            unique_source_count = len(cluster_sources)
            
            raw_clusters.append({
                "indices": indices,
                "evidence_count": evidence_count,
                "has_wiki": has_wiki,
                "has_gg": has_gg
            })

            # ===== PRIMARY RULE =====
            # if not (evidence_count >= 3 and has_wiki):
            if not (evidence_count > 3 or unique_source_count > 6):
                continue

            cohesion = cluster_cohesion(embeddings[indices])

            if cohesion < best_cohesion:
                best_cluster = indices
                best_cohesion = cohesion

        

        # ===== FALLBACK =====
        if best_cluster is None and len(raw_clusters) == 2:

            c1, c2 = raw_clusters

            if c1["evidence_count"] != c2["evidence_count"]:

                larger = c1 if c1["evidence_count"] > c2["evidence_count"] else c2

                if larger["has_wiki"]:
                    best_cluster = larger["indices"]

        # ===== SINGLE CLUSTER =====
        if best_cluster is None and len(raw_clusters) == 1:

            only_cluster = raw_clusters[0]

            if only_cluster["evidence_count"] == 2 and only_cluster["has_wiki"]:
                best_cluster = only_cluster["indices"]

        if best_cluster is None:
            fused_results[qidx] = {"abstain": True}
            continue

        # ===== FINAL SELECTION =====
        centroid = embeddings[best_cluster].mean(axis=0)

        distances = {
            i: np.linalg.norm(embeddings[i] - centroid)
            for i in best_cluster
        }

        min_dist = min(distances.values())

        def candidate_score(i):

            dist = distances[i]

            if not USE_SOURCE_BONUS:
                return dist

            if dist - min_dist > SOURCE_DISTANCE_MARGIN:
                return dist

            has_wiki, has_gg, _ = detect_source_from_long_answer(
                cand_list[i]["long_answer"]
            )

            if has_gg:
                return dist - GG_BONUS
            elif has_wiki:
                return dist - WIKI_BONUS
            else:
                return dist

        best_idx = min(best_cluster, key=candidate_score)

        fused_results[qidx] = {
            "final_answer": cand_list[best_idx]["text"],
            "context": cand_list[best_idx]["context"],
            "abstain": False
        }

    return fused_results


# =========================
# MODEL
# =========================
def load_pubmed_model():
    static = StaticEmbedding.from_model2vec(
        "neuml/pubmedbert-base-embeddings-8M"
    )
    return SentenceTransformer(modules=[static])


# =========================
# GOLDEN EVALUATION
# =========================
def compute_selected_summary(fused_results, dict_golden):

    total_with_evidence = len(fused_results)

    selected = 0
    correct_selected = 0
    total_wrong = 0
    wrong_has_golden_in_ctx = 0
    wrong_no_golden_in_ctx = 0

    for qidx, info in fused_results.items():

        if info.get("abstain", True):
            continue

        selected += 1

        pred = info.get("final_answer", "").lower()
        context = info.get("context", "").lower()

        golden_list = dict_golden.get(qidx, {}).get("True", [])

        golden_list = [g.lower() for g in golden_list]

        is_correct = pred in golden_list

        if is_correct:
            correct_selected += 1
        else:
            total_wrong += 1

            found_in_ctx = any(g in context for g in golden_list)

            if found_in_ctx:
                wrong_has_golden_in_ctx += 1
            else:
                wrong_no_golden_in_ctx += 1

    acc_selected = correct_selected / selected if selected else 0.0
    coverage = selected / total_with_evidence if total_with_evidence else 0.0

    return {
        "total_with_evidence": total_with_evidence,
        "selected": selected,
        "correct_selected": correct_selected,
        "acc_selected": acc_selected,
        "coverage": coverage,
        "total_wrong": total_wrong,
        "wrong_has_golden_in_ctx": wrong_has_golden_in_ctx,
        "wrong_no_golden_in_ctx": wrong_no_golden_in_ctx
    }


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("Loading embedding model...")
    model = load_pubmed_model()

    s_type = "kmean"
    n_cluster = 7

    cluster_file = fr"src\cluster\final\concat_answer_{s_type}_merge_google_wiki_{n_cluster}.json"
    output_path = r"src\cluster\final\fused_results.json"
    dict_golden = load_json("data/all_answer_golden_56.json")

    print("Running clustering + fusion...")
    fused_results = cluster_and_evaluate(
        cluster_file=cluster_file,
        model=model,
    )

    print("Saving results...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fused_results, f, indent=4, ensure_ascii=False)

    print(f"Total questions processed: {len(fused_results)}")

    selected_summary = compute_selected_summary(fused_results, dict_golden)

    print("\n=== SELECTED ANSWER STATISTICS ===")
    print(f"Tổng số câu có evidence   : {selected_summary['total_with_evidence']}")
    print(f"Số câu được chọn          : {selected_summary['selected']}")
    print(f"Coverage                  : {selected_summary['coverage']*100:.2f}%")
    print(f"Số câu đúng               : {selected_summary['correct_selected']}")
    print(f"ACC trên câu được chọn    : {selected_summary['acc_selected']*100:.2f}%")
    print(f"Số câu sai                : {selected_summary['total_wrong']}")
    print(f"   - Có golden trong context    : {selected_summary['wrong_has_golden_in_ctx']}")
    print(f"   - Không có golden trong ctx  : {selected_summary['wrong_no_golden_in_ctx']}")
