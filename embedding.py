from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import os 
import json

device = "cpu"

embedding_model = SentenceTransformer(
    "pritamdeka/S-BioBert-snli-multinli-stsb",
    device=device
)

def get_embedding(text):
    embedding = embedding_model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True  
    )
    return embedding

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
get_crawl_wiki = True
if get_crawl_wiki:
    # Crawl Wiki
    cache_crawl_wiki = r"data\cache\hash_id_crawl_wiki.json"
    folder_filter_crawl_wiki = "data/cache/filter_wikipedia_300"
    output_embed_file = "data/cache/embed_filter_crawl_wiki_300.json"
else:
    # crawl ner wiki
    cache_crawl_wiki = r'data\cache\mapping\url2hash.json'
    folder_filter_crawl_wiki = "data/cache/filter_wikipedia_ner_300"
    output_embed_file = "data/cache/embed_filter_crawl_wiki_300.json"

dict_embed_filter = {}
save_every = 50

if isinstance(cache_crawl_wiki, str):
    cache_crawl_wiki = load_json(cache_crawl_wiki)

for idx, (url, hash_id) in enumerate(cache_crawl_wiki.items(), 1):
    tfidf_file = f"{folder_filter_crawl_wiki}/{hash_id}"
    if os.path.exists(tfidf_file):
        data = load_json(tfidf_file)
        filter_data = data['crawl_wiki']['filter'][url]
        if filter_data:
            embed_filter = get_embedding(filter_data['tfidf'])
            dict_embed_filter[hash_id] = embed_filter
        if idx % save_every == 0:
                # Convert all ndarray to list before saving
                dict_to_save = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dict_embed_filter.items()}
                save_json(dict_to_save, output_embed_file)

        # Save remaining if not already saved
    if len(dict_embed_filter) % save_every != 0:
        dict_to_save = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dict_embed_filter.items()}
        save_json(dict_to_save, output_embed_file)

save_json(dict_embed_filter, output_embed_file)
        
        
