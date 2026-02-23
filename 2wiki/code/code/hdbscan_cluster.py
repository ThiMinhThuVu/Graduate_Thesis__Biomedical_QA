import json
import numpy as np
import hdbscan 
from sklearn.preprocessing import normalize


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# ==========================================================
# HDBSCAN version cho crawl_wiki
# ==========================================================
def cluster_per_qid_crawl_wiki_hdbscan(
        meta,
        cache_hashid,
        cache_crawl,
        cache_crawl_ner,
        min_cluster_size=2,
        min_samples=1):

    results = {}

    for qidx, item in meta.items():
        if item['type_q'] != 'direct' or item['target_answer_type'] == 'correctness':
            continue

        all_embeddings = []
        doc_sources = []

        # ===== 1. crawl_wiki =====
        for hashid, url in cache_hashid[qidx]['crawl_wiki'].items():
            if hashid in cache_crawl:
                emb = cache_crawl[hashid]
                all_embeddings.append(emb)
                doc_sources.append({
                    "hashid": hashid,
                    "url": url,
                    "source": "crawl_wiki"
                })

        # ===== 2. crawl_ner_wiki =====
        for hashid, url in cache_hashid[qidx]['crawl_ner_wiki'].items():
            if hashid in cache_crawl_ner:
                emb = cache_crawl_ner[hashid]
                all_embeddings.append(emb)
                doc_sources.append({
                    "hashid": hashid,
                    "url": url,
                    "source": "crawl_ner_wiki"
                })

        if len(all_embeddings) == 0:
            continue

        embeddings = np.array(all_embeddings)
        embeddings = normalize(embeddings)

        if len(embeddings) < 2:
            results[qidx] = {
                "labels": [0] * len(embeddings),
                "cluster_centers": [embeddings.mean(axis=0).tolist()],
                "docs": doc_sources
            }
            continue

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean'
        )

        labels = clusterer.fit_predict(embeddings)

        # ===== tính center cho từng label (kể cả -1) =====
        unique_labels = sorted(set(labels))
        cluster_centers = []

        for lab in unique_labels:
            points = embeddings[labels == lab]
            center = points.mean(axis=0)
            cluster_centers.append(center.tolist())

        results[qidx] = {
            "labels": labels.tolist(),
            "cluster_centers": cluster_centers,
            "docs": doc_sources
        }

    return results


# ==========================================================
# HDBSCAN version cho search_api
# ==========================================================
def cluster_per_qid_search_api_hdbscan(
        meta,
        cache_article,
        min_cluster_size=3,
        min_samples=None):

    results = {}

    for qidx, item in meta.items():
        if item['type_q'] != 'direct' or item['target_answer_type'] == 'correctness':
            continue

        if qidx not in cache_article:
            continue

        all_embeddings = cache_article[qidx]['embed_docs']
        doc_sources = cache_article[qidx]['docs']

        if len(all_embeddings) == 0:
            continue

        embeddings = np.array(all_embeddings)
        embeddings = normalize(embeddings)

        if len(embeddings) < 2:
            results[qidx] = {
                "labels": [0] * len(embeddings),
                "cluster_centers": [embeddings.mean(axis=0).tolist()],
                "docs": doc_sources
            }
            continue

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            # min_samples=min_samples,
            metric='euclidean'
        )

        labels = clusterer.fit_predict(embeddings)

        unique_labels = sorted(set(labels))
        cluster_centers = []

        for lab in unique_labels:
            points = embeddings[labels == lab]
            center = points.mean(axis=0)
            cluster_centers.append(center.tolist())

        results[qidx] = {
            "labels": labels.tolist(),
            "cluster_centers": cluster_centers,
            "docs": doc_sources
        }

    return results

from collections import defaultdict
import matplotlib.pyplot as plt

def cluster_and_plot_single_qidx(
        meta,
        cache_article,
        target_qidx,
        min_cluster_size=2,
        min_samples=1):

    if target_qidx not in meta:
        print("qidx not in meta")
        return

    if target_qidx not in cache_article:
        print("qidx not in cache_article")
        return

    item = meta[target_qidx]

    if item['type_q'] != 'direct' or item['target_answer_type'] == 'correctness':
        print("qidx not valid type")
        return

    embeddings = np.array(cache_article[target_qidx]['embed_docs'])
    docs = cache_article[target_qidx]['docs']

    if len(embeddings) == 0:
        print("No embeddings")
        return

    embeddings = normalize(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )

    labels = clusterer.fit_predict(embeddings)
    probabilities = clusterer.probabilities_

    print(f"Unique labels: {set(labels)}")
    print(f"Noise count (-1): {(labels == -1).sum()}")

    # ==============================
    # 1️⃣ Scatter plot (PCA 2D)
    # ==============================
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure()

    unique_labels = sorted(set(labels))

    for lab in unique_labels:
        mask = labels == lab

        if lab == -1:
            # Noise → màu đỏ
            plt.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                color='red',
                # label='Noise (-1)',
                alpha=0.9
            )
        else:
            # Cluster bình thường
            plt.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                # label=f'Cluster {lab}',
                alpha=0.7
            )

    plt.title(f"Cluster visualization for qidx {target_qidx}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()
    
    return {
        "labels": labels,
        "probabilities": probabilities,
        "docs": docs
    }
# ==========================================================
# MAIN
# ==========================================================
meta = load_json('data/metadata_medhop.json')
cache_crawl = load_json(r'data\cache\embed_filter_crawl_wiki_300.json')
cache_crawl_ner = load_json(r'data\cache\embed_filter_crawl_wiki_ner_300.json')
cache_hashid = load_json(r'data\cache\all_hash_id_crawl_wiki_embed.json')

source = 'merge_google_wiki' # 'google'
cache_article = load_json(fr'src\cluster\cache\embed_ctx_{source}.json')

crawl_wiki = False

if crawl_wiki:
    cluster_results = cluster_per_qid_crawl_wiki_hdbscan(
        meta,
        cache_hashid,
        cache_crawl,
        cache_crawl_ner,
        min_cluster_size=2,
        min_samples=1
    )

    save_json(cluster_results,
              r"src\cluster\cache\cluster_results_hdbscan_crawl.json")

    print(r"src\cluster\cache\cluster_results_hdbscan_crawl.json")

else:
    # cluster_results = cluster_per_qid_search_api_hdbscan(
    #     meta,
    #     cache_article,
    #     min_cluster_size=3,
    #     min_samples=2
    # )
    min_cluster_size = 2
    min_samples = "none"
    cluster_results = cluster_per_qid_search_api_hdbscan(
        meta,
        cache_article,
        min_cluster_size=min_cluster_size,
        # min_samples=min_samples
    )

    save_json(cluster_results,
              fr"src\cluster\cache\cluster_results_hdbscan_{source}_{min_cluster_size}_{min_samples}.json")

    print(fr"src\cluster\cache\cluster_results_hdbscan_{source}_{min_cluster_size}_{min_samples}.json")
    # cluster_and_plot_single_qidx(
    #     meta,
    #     cache_article,
    #     target_qidx='524762',
    #     min_cluster_size=3,
    #     min_samples=2)
