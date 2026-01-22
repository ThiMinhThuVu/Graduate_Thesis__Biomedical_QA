import requests
import json

def search_google(query):
    url = "https://api.serpshot.com/api/search/google"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "a631791792b638a6cf7382b9c947c167"
    }

    payload = {
        "queries": [query],
        "type": "search",
        "lr": "en",
        "gl": "us",
        "hl": "en",
        "location": None,
        "num": 10,
        "page": 1
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    result = response.json().get('data')[0].get('results')
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    with open('google_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def _search_google(query):
    url = "https://www.searchapi.io/api/v1/search"
    params = {
    "engine": "google",
    "q": query,
    "api_key": "cJTZKyQkKAT5qZpCWoAhDua7"
    }

    response = requests.get(url, params=params)
    result = response.json()
    print(response.text)
    with open('google_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

query = "medication powdered mineral treat itchiness contact dermatitis toxicodendron poison ivy"
search_google(query)