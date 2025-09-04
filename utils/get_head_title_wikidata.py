import requests
import re
import time
from bs4 import BeautifulSoup
from utils.get_abs_wiki_history import get_header_wikipedia_history
    
def get_header_title_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Kiểm tra lỗi HTTP

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Tìm thẻ <span> với class là "wikibase-title-label"
        # Đây là nơi chứa label của thực thể
        label_span = soup.find('span', class_='wikibase-title-label')
        
        if label_span:
            # Lấy nội dung văn bản của thẻ <span>
            return label_span.get_text(strip=True)
        
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi truy cập URL: {e}")
        if e.response.status_code == 429:
             print("Too many requests. Waiting for 60 seconds...")
             time.sleep(60)        
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")        
    return None

def get_wikidata_history_info(qid, date_str="2021-04-07"):
    if "https://en.wikipedia.org/wiki" in qid:
        return get_header_wikipedia_history(qid)
    
    # The Wikidata API endpoint
    if "http" in qid:
        qid = qid.split('/')[-1]
        
    URL = "https://www.wikidata.org/w/api.php"

    # The revisions query parameters, including 'redirects'
    params_revisions = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": qid,
        "rvprop": "ids|timestamp|user|comment",
        "rvdir": "older", # Find the last revision before the start date
        "rvstart": f"{date_str}T00:00:00Z",
        "rvlimit": 1,
        "redirects": 1 # This is crucial for handling redirects
    }
    
    headers = {
        # "User-Agent": "MyWikidataApp/1.0"
        "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        res = requests.get(URL, params=params_revisions, headers=headers)
        res.raise_for_status()
        data = res.json()
        time.sleep(1) 
        pages = data.get("query", {}).get("pages", {})
        
        # Check if the QID exists and has revisions before the date
        if not pages or '-1' in pages:
            print("No revisions found for this QID before the specified date.")
            return None

        page_id = next(iter(pages.keys()))
        revision_data = pages[page_id].get("revisions", [])

        if not revision_data:
            print("No revisions found for this QID before the specified date.")
            return None

        # Get the latest revision ID and timestamp
        revision = revision_data[0]
        revision_id = revision.get("revid")
        timestamp = revision.get("timestamp")
        
        print(f"Found revision ID {revision_id} from {timestamp}")

        # Get the final QID after any redirects
        final_qid = pages[page_id].get("title")
        new_url = f"https://www.wikidata.org/w/index.php?title={final_qid}&oldid={revision_id}"
        header_title = get_header_title_from_url(new_url)
        
        return header_title
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    except KeyError as e:
        print(f"An error occurred while parsing JSON: {e}")
        return None

if __name__ == "__main__":
    # Example usage:
    qid_to_check = "Q155734" # A QID that redirects
    header_title = get_wikidata_history_info(qid_to_check)

    print(header_title)