import requests
import time
from src.api_key import API_LIST_4O, API_LIST_O3, API_KEY_GG

API_URL = "https://cf.mnapi.com/v1/chat/completions"  

current_api_index = 0

def request_api(messages, model, temperature=0):
    global current_api_index
    max_retries = 10
    retry_delay = 1
    
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

def search_google(query, before_date="2021-04-07"):
    max_retries = 10
    retry_delay = 2
    
    if before_date:
        query = f"{query} before:{before_date}"
        
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

if __name__ == "__main__":    
    # result = search_google("Is cabergoline safe to use for a patient with cholestasis who has a prolactinoma?")
    result = request_api(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        model="o3-mini"
    )
    print(result)
