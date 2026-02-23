import json
from sklearn.cluster import KMeans
import numpy as np

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def cluster_per_qid_crawl_wiki(meta, cache_hashid, cache_crawl, cache_crawl_ner, n_clusters=3):
    results = {}

    for qidx, item in meta.items():
        if item['type_q'] != 'direct' or item['target_answer_type'] != 'correctness': 
            continue

        all_embeddings = []
        doc_sources = []  # để giữ mapping doc nào thuộc cluster nào

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

        # nếu không có doc → bỏ qua
        if len(all_embeddings) == 0:
            continue

        embeddings = np.array(all_embeddings)

        # giảm K nếu N nhỏ
        k = min(n_clusters, len(embeddings))

        if k < 2:
            results[qidx] = {
                "labels": [0] * len(embeddings),
                "cluster_centers": [embeddings.mean(axis=0).tolist()],
                "docs": doc_sources
            }
            continue

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(embeddings)

        results[qidx] = {
            "labels": labels.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "docs": doc_sources
        }

    return results

def cluster_per_qid_search_api(meta, cache_article, n_clusters=3):
    results = {}
    for qidx, item in meta.items():
        if item['type_q'] != 'direct' or item['target_answer_type'] != 'correctness': 
            continue
        
        all_embeddings = cache_article[qidx]['embed_docs']
        doc_sources = cache_article[qidx]['docs']

        if len(all_embeddings) == 0:
            continue

        embeddings = np.array(all_embeddings)

        # giảm K nếu N nhỏ
        k = min(n_clusters, len(embeddings))

        if k < 2:
            results[qidx] = {
                "labels": [0] * len(embeddings),
                "cluster_centers": [embeddings.mean(axis=0).tolist()],
                "docs": doc_sources
            }
            continue


        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(embeddings)

        results[qidx] = {
            "labels": labels.tolist(),
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "docs": doc_sources
        }

    return results

meta = load_json('data/metadata_medhop.json')
cache_crawl = load_json(r'data\cache\embed_filter_crawl_wiki_300.json')
cache_crawl_ner = load_json(r'data\cache\embed_filter_crawl_wiki_ner_300.json')
cache_hashid = load_json(r'data\cache\all_hash_id_crawl_wiki_embed.json')

source = 'merge_google_wiki'
n_clusters = 7 #3
crawl_wiki = False
cache_article = load_json(fr'data\cache\embed_ctx_{source}_YN.json')

if crawl_wiki:
    #CRAWL_WIKI
    cluster_results = cluster_per_qid_crawl_wiki(
        meta,
        cache_hashid,
        cache_crawl,
        cache_crawl_ner,
        n_clusters=n_clusters
    )
    save_json(cluster_results, fr"src\cluster\cache\kmean\cluster_results_300_{n_clusters}.json")
    print(fr"src\cluster\cache\kmean\cluster_results_300_{n_clusters}.json")
else:
    cluster_results = cluster_per_qid_search_api(meta, cache_article, n_clusters=n_clusters)
    save_json(cluster_results, fr"src\cluster\cache\kmean\YN_cluster_results_kmean_{source}_{n_clusters}.json")
    print(fr"src\cluster\cache\kmean\YN_cluster_results_kmean_{source}_{n_clusters}.json")
    

