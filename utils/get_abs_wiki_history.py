import requests
import json
from bs4 import BeautifulSoup


URL = "https://en.wikipedia.org/w/api.php"

def get_revision_id(title):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": "1",
        "rvprop": "ids|timestamp|user|comment",
        "rvdir": "older",
        "rvstart": "2021-04-07T00:00:00Z",
        "format": "json"
    }

    res = requests.get(URL, params=params, headers=headers)
    if res.status_code == 403:
        print("Lỗi: Yêu cầu bị từ chối (403 Forbidden).")
        return None
    
    data = res.json()
    pages = data.get("query", {}).get("pages", {})

    revision_id = None
    for page_id, page_data in pages.items():
        revisions = page_data.get("revisions", [])
        if revisions:
            rev = revisions[0]
            revision_id = rev["revid"]
            print(f"Revision ID: {revision_id}")
            return revision_id
            
    return revision_id

def get_abs_wiki_history(title):
    revision_id = get_revision_id(title)
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

def get_header_wikipedia_history(q_url):
    clean_url = q_url.replace("<", "").replace(">", "")
    title = clean_url.split("/")[-1]
    revision_id = get_revision_id(title)
    if revision_id != None:
        url = f"https://en.wikipedia.org/w/index.php?title={title}&oldid={revision_id}"
    else:
        url = clean_url
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Tiêu đề trên Wikipedia nằm trong thẻ <h1> có id="firstHeading"
        header_tag = soup.find('h1', {'id': 'firstHeading'})
        
        if header_tag:
            return header_tag.get_text(strip=True)
            
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi truy cập URL: {e}")
        
    return None
    