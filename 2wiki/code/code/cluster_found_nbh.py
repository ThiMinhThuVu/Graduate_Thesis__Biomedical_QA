import os
import json
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
from copy import deepcopy 

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def calculate_cosine_similarity(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]

def calculate_centroid(docs):
    matrix = np.array(docs)
    centroid = np.mean(matrix, axis=0)
    return centroid.tolist()

def get_wiki_title(url):
    prefix = "https://en.wikipedia.org/wiki/"
    if url.startswith(prefix):
        return url[len(prefix):]
    return None

def crawl_passages(): # Tổng hợp 2 nguồn crawl
    HASH_ID_PATH = "data/cache/all_hash_id_crawl_wiki_embed.json"
    EMBEDDING_CRAWL_WIKI_PATH = "data/cache/embed_filter_crawl_wiki_300.json"
    EMBEDDING_CRAWL_WIKI_NER_PATH = "data/cache/embed_filter_crawl_wiki_ner_300.json"
    BASE_WIKI = "data/cache/filter_wikipedia_300"
    BASE_WIKI_NER = "data/cache/filter_ner_wikipedia_300"

    hash_id_data = load_json(HASH_ID_PATH)
    embedding_crawl_wiki = load_json(EMBEDDING_CRAWL_WIKI_PATH)
    embedding_crawl_wiki_ner = load_json(EMBEDDING_CRAWL_WIKI_NER_PATH)

    passages = {}
    
    crawl_source = ["crawl_wiki", "crawl_ner_wiki"]

    for qidx in tqdm(hash_id_data, desc="Crawl: "):
        if qidx not in passages:
            passages[qidx] = []

        crawl_wiki_hash = hash_id_data[qidx]["crawl_wiki"]
        crawl_wiki_ner_hash = hash_id_data[qidx]["crawl_ner_wiki"]
        source_configs = {
            "crawl_wiki": {
                "embedding_dict": embedding_crawl_wiki,
                "base_path": BASE_WIKI,
                "hash_map": crawl_wiki_hash
            },
            "crawl_ner_wiki": {
                "embedding_dict": embedding_crawl_wiki_ner,
                "base_path": BASE_WIKI_NER,
                "hash_map": crawl_wiki_ner_hash
            }
        }
        for source_name in crawl_source:
            config = source_configs.get(source_name)
            if not config:
                continue

            for hash_id in config["hash_map"]: 
                if hash_id in config["embedding_dict"]:
                    embedding = config["embedding_dict"][hash_id]
                    url = config["hash_map"][hash_id]
                    title = get_wiki_title(url)
                    file_path = os.path.join(config["base_path"], hash_id)
                    filter_text = load_json(file_path)
                    
                    content = f"- Information about: {title}\n" +  filter_text.get("crawl_wiki", {}).get("filter", {}).get(url, {}).get("tfidf", None)
                    
                    passages[qidx].append({
                        'docs': content,
                        'embedding': embedding,
                        "hash_id": hash_id,
                        "source": source_name
                    })
    return passages

def get_top_k_docs(centroid, crawl_data, topk=5, method='similarity'): # lấy top k từ tâm cụm search    
    if not crawl_data:
        print("⚠ Empty crawl_data")
        return [], [], []

    doc_embeddings = [doc['embedding'] for doc in crawl_data]

    if len(doc_embeddings) == 0:
        print("⚠ No embeddings found")
        return [], [], []

    doc_matrix = np.array(doc_embeddings)

    if doc_matrix.ndim != 2:
        print("⚠ doc_matrix not 2D:", doc_matrix.shape)
        return [], [], []

    centroid = np.array(centroid).reshape(1, -1)
      
    if method == 'euclidean':
        distances = cdist(centroid, doc_matrix, metric='euclidean')[0]
        top_indices = np.argsort(distances)[:topk]
        top_scores = distances[top_indices]
        
    elif method == 'similarity':
        cosine_dist = cdist(centroid, doc_matrix, metric='cosine')[0]
        similarities = 1 - cosine_dist
        top_indices = np.argsort(similarities)[::-1][:topk]
        top_scores = similarities[top_indices]
        
    else:
        raise ValueError("Method phải là 'euclidean' hoặc 'similarity'")
        
    top_texts = [crawl_data[i]['docs'] for i in top_indices]
    top_hash_ids = [crawl_data[i]['hash_id'] for i in top_indices]
    top_sources = [crawl_data[i]['source'] for i in top_indices]

    return top_texts, top_hash_ids, top_sources

import numpy as np

def compute_cluster_info(labels, docs, embed_docs):
    """
    Gom doc + embedding theo từng label (bao gồm cả -1) và tính centroid.

    Args:
        labels (list[int])
        docs (list[str])
        embed_docs (list[list[float]])

    Returns:
        dict[label] = {
            "centroid": list[float],
            "docs": list[str],
            "embeddings": list[list[float]]
        }
    """

    assert len(labels) == len(docs) == len(embed_docs), \
        "Labels, docs và embeddings không cùng độ dài!"

    cluster_data = {}

    for label, doc, emb in zip(labels, docs, embed_docs):

        if label not in cluster_data:
            cluster_data[label] = {
                "docs": [],
                "embeddings": []
            }

        cluster_data[label]["docs"].append(doc)
        cluster_data[label]["embeddings"].append(emb)

    for label in cluster_data:
        vectors = np.array(cluster_data[label]["embeddings"])

        if vectors.ndim == 1:
            centroid = vectors
        else:
            centroid = np.mean(vectors, axis=0)

        # Convert sang list để save JSON
        cluster_data[label]["centroid"] = centroid.tolist()

    return cluster_data


if __name__ == "__main__":
    topk = 5 #5
    cluster_type = 'kmean' #'kmean'  #'hdbscan'
    methods = ['euclidean', 'similarity']
    method = methods[0]

    n_cluster = "7"
    # YESNO
    cluster_search_file = r'E:\MAS_UETQuintet\src\cluster\final\cluster_results_kmean_merge_google_wiki_7.json'    
    file_output = f"src/cluster/final/direct_output_cluster_{cluster_type}_{method}_top{topk}_{n_cluster}.json"
    embed_ctx_gg_wiki = r"data\cache\embed_ctx_merge_google_wiki_direct.json"


    #cluster_search_file=fr'src\cluster\cache\{cluster_type}\YN_cluster_results_{cluster_type}_merge_google_wiki_{n_cluster}.json' 
    # file_output = f"src/cluster/final/YN_output_cluster_{cluster_type}_{method}_top{topk}_{n_cluster}.json"
    # embed_ctx_gg_wiki = "data\cache\embed_ctx_merge_google_wiki_YN.json"

    # cluster_search_file = fr'src\cluster\cache\{cluster_type}\cluster_results_{cluster_type}_merge_google_wiki_{n_cluster}.json'
    # file_output = f"src/cluster/final/output_cluster_{cluster_type}_{method}_top{topk}_{n_cluster}.json"    
    # embed_ctx_gg_wiki = "data\cache\embed_ctx_merge_google_wiki.json"
    
    cluster_search = load_json(cluster_search_file)
    embed_doc_search = load_json(embed_ctx_gg_wiki)
    
    dict_crawl_data = crawl_passages()

    output_dict = {}

    for qidx in tqdm(cluster_search, desc="Top k: "):
        output_dict[qidx] = {}

        labels = cluster_search[qidx]["labels"]
        docs = cluster_search[qidx]["docs"]
        embed_docs = embed_doc_search[qidx]["embed_docs"]

        cluster_info = compute_cluster_info(labels, docs, embed_docs)

        for cluster_label, info in cluster_info.items():
            centroid = info["centroid"]
            cluster_docs = info["docs"]

            crawl_docs, top_hash_ids, top_sources = get_top_k_docs(
                centroid=centroid,
                crawl_data=dict_crawl_data[qidx],
                topk=topk,
                method=method
            )

            output_dict[qidx][cluster_label] = {
                "docs": cluster_docs,
                "crawl_doc": crawl_docs,
                'hash_id': top_hash_ids,
                'source': top_sources
            }

    save_json(
        output_dict,
        file_output
    )

    print(file_output)