import hashlib

def normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())

def hash_query(query: str, hash_len: int = 32) -> str:
    """
    Hash a query string into a stable ID
    """
    normalized = normalize_query(query)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest[:hash_len]   # rút gọn cho tên file

def make_cache_filename(query: str) -> str:
    query_id = hash_query(query)
    return f"{query_id}.json"

cache_index = {}

def register_query_cache(query: str, cache_index: dict) -> str:
    """
    Register query into cache dict.
    Return cache filename.
    """
    if query in cache_index:
        return cache_index[query]

    filename = make_cache_filename(query)
    cache_index[query] = filename
    return filename

if __name__ == "__main__":
    query = "chromosome location gene responsible Carpenter syndrome MEGF8, RAB23"
    cache_index = {}
    cache_file = register_query_cache(query, cache_index)

    print(cache_file)
    print(cache_index)
