import requests
import time
import json

from src.api_key import API_LIST_4O_mini, API_LIST_O3_mini,API_LIST_4_1_mini, API_SEARCH_1, API_LIST_O3, request_api_o3_mini

API_URL = "https://cf.mnapi.com/v1/chat/completions"  

current_api_index = 0

def request_api(messages, model, temperature=0):
    global current_api_index
    max_retries = 50
    retry_delay = 10
    
    if model == "gpt-4o-mini":
        API_LIST = API_LIST_4O_mini
        max_tokens = 4096
    elif model == "o3-mini":
        max_tokens = 8192
        API_LIST = API_LIST_O3_mini
    elif model == "gpt-4.1-mini":
        API_LIST = API_LIST_4_1_mini
        max_tokens = 8192
    elif model == "o3":
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

# def search_1_search_google(query):
# def search_google(query, max_results=10):
    max_retries = 10
    retry_delay = 2
   
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'https://api.search1api.com/search',
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': API_SEARCH_1 
                },
                json={
                    "query": query,
                    "search_service": "google",
                    "max_results": max_results,
                    "crawl_results": 0,
                    "image": False,
                    "language": "en"
                },
                timeout=30
            )

            if response.status_code in [200, 404, 502]:
                return response.json().get("results", [])
            else:
                raise RuntimeError(f"Search API failed with status code {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise e  

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
import http.client
import json as _json
from urllib.parse import urlencode

conn = http.client.HTTPSConnection("scraperapi.thordata.com")
    
API_search_scraperapi = '7b29596f5d6ce1f7c5b5aeeb96f60e72' #'fa1e1e5253fadc4cdaef9f5a4f7ea85e' #'f9d4ae75aaa264fcb318051f9fd17aff'
# def scraperapi_search(query):
def search_google(query, max_results=20):  
    params = {
        "engine": "google",
        "q": query,
        "json": "1",
        "num": max_results  
    }

    payload = urlencode(params)

    headers = {
        'Authorization': f'Bearer {API_search_scraperapi}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    try:
        conn.request("POST", "/request", payload, headers)
        res = conn.getresponse()

        if res.status in (401, 403):
            print("API key không hợp lệ hoặc đã hết hạn (status:", res.status, ")")
            return None

        data = res.read()
        data = _json.loads(data.decode("utf-8"))

        output = []

        # ===== LẤY TỐI ĐA max_results =====
        organic_results = data.get('organic', [])[:max_results]

        for item in organic_results:
            output.append({
                "title": item.get('title', ''),
                "link": item.get('link', ''),
                "snippet": item.get('description', '')
            })

        return output

    except Exception as e:
        print("search_google error:", e)
        return None
    
if __name__ == "__main__":    
    # Ví dụ hỏi thủ đô của Pháp bằng model o3
    messages = [
        {"role": "user", "content": "Thủ đô của Pháp là gì?"}
    ]
    print(request_api(messages, model="o3-mini"))