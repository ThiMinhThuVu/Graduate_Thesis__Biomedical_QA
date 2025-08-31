import requests

URL = "https://en.wikipedia.org/w/api.php"

def get_abs_wiki_history(title):
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": "1",
        "rvprop": "ids|timestamp|user|comment",
        "rvdir": "older",   # từ mới -> cũ
        "rvstart": "2021-04-07T00:00:00Z",  # bắt đầu từ mốc 7/4/2021
        "format": "json"
    }

    res = requests.get(URL, params=params)
    data = res.json()

    pages = data["query"]["pages"]

    revision_id = None
    for page_id, page_data in pages.items():
        revisions = page_data.get("revisions", [])
        if revisions:
            rev = revisions[0]
            revision_id = rev["revid"]
            print(f"Revision ID: {revision_id}")
            # print(f"Timestamp: {rev['timestamp']}")
            # print(f"User: {rev['user']}")
            # print(f"Comment: {rev.get('comment', '')}")

    if revision_id:
        params_extract = {
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "revids": revision_id,
            "format": "json"
        }
        res2 = requests.get(URL, params=params_extract)
        data2 = res2.json()

        pages2 = data2["query"]["pages"]
        for page_id, page_data in pages2.items():
            # print("\n--- ABSTRACT ---")
            # print(page_data.get("extract", ""))
            return page_data.get("extract", "")
