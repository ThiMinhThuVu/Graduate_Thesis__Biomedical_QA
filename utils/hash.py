import hashlib
from utils.util import save_json,load_json

def normalize_query(query: str) -> str:
    return query.lower().strip()

def hash_query(query: str, hash_len: int = 32) -> str:
    """
    Hash a query string into a stable ID
    """
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return digest[:hash_len]   # rút gọn cho tên file

def make_cache_filename(query: str) -> str:
    query_id = hash_query(query)
    return f"{query_id}.json"

def register_query_cache(query: str, cache_index: dict, file_cache_index="data/cache/hash_id.json") -> str:
    """
    Register query into cache dict.
    Return cache filename.
    """
    # query = normalize_query(query.lower())
    query = query.lower().strip()
    if query in cache_index:
        return cache_index[query], cache_index

    filename = make_cache_filename(query)
    cache_index[query] = filename
    save_json(cache_index, file_cache_index)
    
    return filename, cache_index

def register_query_cache_2(query: str, file_cache_index="data/cache/hash_id.json") -> str:
    """
    Register query into cache dict.
    Return cache filename.
    """
    cache_index = load_json(file_cache_index)
    query = normalize_query(query.lower())
    if query in cache_index:
        return cache_index[query]

    filename = make_cache_filename(query)
    cache_index[query] = filename
    save_json(cache_index, file_cache_index)
    
    return filename

if __name__ == "__main__":
    query = "human chromosome location gene PCDH15"
    # cache_file = r"data\cache\hash_id copy.json"
    # cache_index = {}
    # cache_file, cache_index = register_query_cache(query, cache_index)

    # print(cache_file)
    # print(cache_index)
    print(make_cache_filename(query.lower()))