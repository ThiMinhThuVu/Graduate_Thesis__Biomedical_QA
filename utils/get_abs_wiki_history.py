import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote
import re

URL = "https://en.wikipedia.org/w/api.php"
headers = {
        "User-Agent": "WikiRevisionTester/1.0 (contact: 22028116@vnu.edu.vn)",
        "Accept": "application/json"
    }

def get_revision_id(title, bf_ts = "2021-04-07T00:00:00Z"):
    
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "prop": "revisions",
        "rvlimit": "1",
        "rvprop": "ids|timestamp|user|comment",
        "rvdir": "older",
        "rvstart": bf_ts,
        "format": "json",
        "formatversion": 2
    }

    res = requests.get(URL, params=params, headers=headers, timeout=15)
    if res.status_code == 403:
        print("Lỗi: Yêu cầu bị từ chối (403 Forbidden).")
        return None
    if res.status_code != 200:
        print(f"Lỗi: HTTP {res.status_code} khi lấy revision id cho '{title}'")
        return None

    try:
        data = res.json()
    except (ValueError, json.JSONDecodeError):
        print(f"Lỗi: Không thể phân tích JSON trả về khi lấy revision id cho '{title}'. Response length={len(res.text) if res.text is not None else 0}")
        return None
    
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        print("No pages returned")
        return None

    page = pages[0]
    revisions = page.get("revisions", [])

    if not revisions:
        print("No revisions found")
        return None

    rev = revisions[0]
    print("Revision found:")
    print(json.dumps(rev, indent=2))

    return rev["revid"]

def is_bad_wiki_url(title_or_url: str) -> bool:
    """
    Trả về True nếu title / URL không nên dùng để lấy abstract / revision
    """

    if not title_or_url:
        return True

    text = unquote(title_or_url).lower()

    # Category, Help, File, Template...
    bad_prefixes = [
        "category:",
        "file:",
        "help:",
        "template:",
        "portal:",
        "special:",
        "wikipedia:"
    ]
    if any(text.startswith(p) for p in bad_prefixes):
        return True

    # Wikidata / Q-id redirect
    bad_patterns = [
        "/wiki/d:",
        "wikidata.org/wiki/q",
        "wikidata.org/entity/",
    ]
    if any(p in text for p in bad_patterns):
        return True

    return False

def extract_abstract_from_wikitext(wikitext: str) -> str:
    if not wikitext:
        return ""

    lead = wikitext.split("\n==", 1)[0]

    lead = re.sub(r"\{\{.*?\}\}", "", lead, flags=re.DOTALL)

    lead = re.sub(r"<ref.*?>.*?</ref>", "", lead, flags=re.DOTALL)

    lead = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", lead)

    lead = lead.replace("'''", "").replace("''", "")

    lead = re.sub(r"\n{2,}", "\n", lead)
    lead = lead.strip()

    return lead

def get_abs_wiki_history(title):
    if is_bad_wiki_url(title):
        print(f"Skip bad wiki title/url: {title}")
        return ""
    
    revision_id = get_revision_id(title)
    if not revision_id:
        return ""

    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "revids": revision_id,
        "rvslots": "main",
        "rvprop": "content",
        "formatversion": 2
    }

    res = requests.get(URL, params=params, headers=headers, timeout=15)

    if res.status_code != 200:
        print(f"HTTP {res.status_code} khi lấy revision content ({revision_id})")
        print(res.text[:300])
        return ""

    try:
        data = res.json()
    except json.JSONDecodeError:
        print("JSON decode error")
        return ""

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return ""

    revisions = pages[0].get("revisions", [])
    if not revisions:
        return ""

    raw_wikitext = revisions[0]["slots"]["main"]["content"]

    abstract = extract_abstract_from_wikitext(raw_wikitext)
    return abstract

def get_header_wikipedia_history(q_url):
    clean_url = q_url.replace("<", "").replace(">", "")
    title = clean_url.split("/")[-1]
    revision_id = get_revision_id(title)
    if revision_id != None:
        url = f"https://en.wikipedia.org/w/index.php?title={title}&oldid={revision_id}"
    else:
        url = clean_url

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
    