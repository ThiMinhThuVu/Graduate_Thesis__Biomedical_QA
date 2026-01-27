import requests
import time
import json

from src.api_key import API_LIST_4O, API_LIST_O3, API_KEY_GG, API_KEY_serp

API_URL = "https://cf.mnapi.com/v1/chat/completions"  

current_api_index = 0

def request_api(messages, model, temperature=0):
    global current_api_index
    max_retries = 50
    retry_delay = 30
    
    if model == "gpt-4o-mini":
        API_LIST = API_LIST_4O
        max_tokens = 4096
    elif model == "o3-mini":
        API_LIST = API_LIST_O3
        max_tokens = 8192
        
    data = {
        "model": model, 
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    for attempt in range(max_retries):
        api = API_LIST[current_api_index]

        headers = {
            "Authorization": f"Bearer {api}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            response = requests.post(API_URL, headers=headers, json=data)
            
            if response.status_code == 200:
                response = response.json()

                if "choices" in response and len(response["choices"]) > 0:
                    result_json = response["choices"][0].get("message", {}).get("content", "")
                    if result_json == "":
                        print(f"Empty response from API {api}. Retrying...")
                    return result_json
                
                else:
                    raise ValueError("Unexpected response format from API: 'choices' not found or empty.")
            else:
                print(f"API {api} failed with status code {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Error with API {current_api_index}: {e}")

        current_api_index = (current_api_index + 1) % len(API_LIST)
        print(f"Retrying with next API in {retry_delay} seconds...")
        time.sleep(retry_delay)

    raise RuntimeError("All APIs failed after multiple retries.")

def _search_google(query):
    max_retries = 10
    retry_delay = 2
   
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'https://api.search1api.com/search',
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': API_KEY_GG 
                },
                json={
                    "query": query,
                    "search_service": "google",
                    "max_results": 10,
                    "crawl_results": 0,
                    "image": False,
                    "language": "en"
                },
                timeout=30
            )

            if response.status_code in [200, 404]:
                return response.json().get("results", [])
            else:
                raise RuntimeError(f"Search API failed with status code {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise e  

def serpshot_search_google(query):
    url = "https://api.serpshot.com/api/search/google"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "a631791792b638a6cf7382b9c947c167"
    }

    payload = {
        "queries": [query],
        "type": "search",
        "lr": "en",
        "gl": "us", #us
        "hl": "en",
        "location": None,
        "num": 10,
        "page": 1
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(response)
    data = response.json().get('data')
    if data and isinstance(data, list) and len(data) > 0:
        return data[0].get('results', [])
    else:
        return []


def serper_search_google(query):
    url = "https://google.serper.dev/search"

    payload = {
        "q": query,
        "gl": 'us'
    }
    headers = {
    'X-API-KEY': '2a07151d86520e144f999ea39bad7ba10bff8d82',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, json=payload)
    data = response.json().get('organic', {})
    # print(data)
    return data

from serpapi import GoogleSearch

def serapi_search_google(
    query: str,
    num_results: int = 10,
    language: str = "en",
    country: str = "us"
):
    """
    Search Google using SerpAPI.

    Args:
        query (str): Search query
        api_key (str): SerpAPI API key
        num_results (int): Number of results to return
        language (str): Language code (default: en)
        country (str): Country code (default: us)

    Returns:
        list[dict]: Search results (title, snippet, link)
    """

    params = {
        "engine": "google",
        "q": query,
        "api_key": API_KEY_serp,
        "hl": language,
        "gl": country,
        "num": num_results
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    output = []
    for item in results.get("organic_results", []):
        output.append({
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "link": item.get("link")
        })

    return output

import http.client
import json as _json
from urllib.parse import urlencode
conn = http.client.HTTPSConnection("scraperapi.thordata.com")

# def scraperapi_search(query):
def search_google(query):    
    params = {
        "engine": "google",
        "q": query,
        "json": "1"
    }
    payload = urlencode(params)

    headers = {
        'Authorization': 'Bearer 5b7ed6242fd639a02538e19efcf51a0d',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    conn.request("POST", "/request", payload, headers)
    res = conn.getresponse()
    data = res.read()
    try:
        data = _json.loads(data.decode("utf-8"))
        output = []
        for item in data.get('organic', []):
            output.append({
                "title": item.get('title', ''),
                'link': item.get('link', ''),
                'snippet': item.get('description', '')
            })
        return output
    except Exception as e:
        print("search_google error:", e)
        return []

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":    
    # save_json(search_google("Is cabergoline safe to use for a patient with cholestasis who has a prolactinoma?"), "f1.json")
    # save_json(serpshot_search_google("Is cabergoline safe to use for a patient with cholestasis who has a prolactinoma?"), "f2.json")
    save_json(search_google("growth-delay syndrome mutations gene"), "f2.json")
    
    question = "Where did that person graduate from?"
    