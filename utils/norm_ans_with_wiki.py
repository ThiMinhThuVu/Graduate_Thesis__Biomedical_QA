import wikipediaapi
import requests
import json
from bs4 import BeautifulSoup  
import pandas as pd
import time 
from tqdm import tqdm
import re

def get_title_1(QUERY):
    wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='thu/1.0 (22028116@vnu.edu.vn)'  
    )

    page = wiki.page(QUERY)

    if page.exists():
        return page.title
    else:
        return None
    
def get_title_2(QUERY):
    API_URL = "https://en.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "thu/1.0 (22028116@vnu.edu.vn)"  
    }

    params = {
        "action": "opensearch",
        "search": QUERY,
        "limit": "5",
        "namespace": "0",
        "format": "json"
    }

    try:
        response = requests.get(url=API_URL, params=params, headers=headers, timeout=10)

        if response.status_code == 200:
            try:
                data = response.json()
            except Exception as e:
                print(f"Error decoding JSON for '{QUERY}': {e}")
                return None
        else:
            print(f"HTTP Error {response.status_code} for query: '{QUERY}'")
            return None

    except Exception as e:
        print(f"Exception during API call for '{QUERY}': {e}")
        return None

    titles = data[1] 
    urls = data[3]    

    if titles:
        for i, (title, url) in enumerate(zip(titles, urls), start=1):
            try:
                page_response = requests.get(url, timeout=10)
                if page_response.status_code == 200:
                    soup = BeautifulSoup(page_response.text, 'html.parser')
                    html_title = soup.title.string.strip() if soup.title else "N/A"
                    parts = html_title.split(" - Wikipedia")
                    return parts[0]      
                else:
                    print(f"Failed to fetch page (status {page_response.status_code})\n")
            except Exception as e:
                print(f"Error fetching page: {e}\n")
    else:
        print(f"No matching articles found for the query: '{QUERY}'")
    return None


def get_title_3(QUERY):
    API_URL = "https://en.wikipedia.org/w/api.php"

    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": QUERY,
        "format": "json"
    }

    try:
        search_response = requests.get(API_URL, params=search_params, timeout=10)

        if search_response.status_code == 200:
            search_data = search_response.json()
        else:
            print(f"HTTP Error {search_response.status_code} for query: '{QUERY}'")
            return None

    except Exception as e:
        print(f"Exception during Wikipedia search for '{QUERY}': {e}")
        return None

    if "query" in search_data and "search" in search_data["query"] and len(search_data["query"]["search"]) > 0:
        results = search_data["query"]["search"]
        PAGE_TITLE = results[0]["title"]
        return PAGE_TITLE
    else:
        print("No relevant pages found for the query.")
        return None
    
def get_title_4(QUERY):
    QUERY = QUERY.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{QUERY}"
    
    try: 
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        header = soup.find('h1', id = 'firstHeading')
        if header:
            return header.text.strip()
        else:
            return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

def is_valid_query(query):
    """
    Trả về False nếu query chỉ chứa số, hoặc là phần trăm (%), phân số (1/2), tỷ lệ (3:1), v.v.
    """
    query = query.strip()
    if query == "":
        return False

    # Kiểm tra nếu chỉ toàn số hoặc số thực
    if re.fullmatch(r"\d+(\.\d+)?", query):
        return False
    
    # Phần trăm: 50%, 100%
    if re.fullmatch(r"\d+(\.\d+)?%", query):
        return False
    
    # Phân số: 1/2, 3/4
    if re.fullmatch(r"\d+/\d+", query):
        return False
    
    # Tỷ lệ: 3:1
    if re.fullmatch(r"\d+:\d+", query):
        return False

    # Nếu query chỉ là ký tự không phải chữ (vd: "#$@!")
    if not re.search(r"[a-zA-Z]", query):
        return False

    return True

def bioasq():
    raw_data_path = r"data\bioasq_12b\output_B2.csv"
    raw_data = pd.read_csv(raw_data_path)
    
    raw_data['title_wiki'] = ''
    for i, row in raw_data.iterrows():
        short_ans = row['short_answers']
        
        if pd.isna(short_ans): 
            raw_data.at[i, 'title_wiki'] = ''
            continue
        
        list_ans = short_ans.split('|')
        
        wiki_titles = []
        for ans in list_ans:
            
            ans = ans.strip()
            if ans:
                print(f"Processing answer: {ans}")
                title = get_title_1(ans)
                if not title:
                    title = get_title_2(ans)
                if not title:
                    title = get_title_3(ans)
                if not title:
                    title = get_title_4(ans)
                    
                if title:
                    print(f"Found title: {title}\n")
                    wiki_titles.append(title)                    
                else:
                    print("No title found.\n")
            time.sleep(3)
        raw_data.at[i, 'title_wiki'] = '|'.join(wiki_titles) if wiki_titles else ''
    
    output_file = r"data\bioasq_12b\output_B2_with_wiki.csv"
    raw_data.to_csv(output_file, index=False)
    print(f"File đã được lưu tại: {output_file}")
    
def medhop():
    raw_data_path = r"data\bioasq_12b\output_B2.csv"
    raw_data = pd.read_csv(raw_data_path)
    
    raw_data['title_wiki'] = ''
    for i, row in raw_data.iterrows():
        short_ans = row['short_answers']
        
        if pd.isna(short_ans): 
            raw_data.at[i, 'title_wiki'] = ''
            continue
        
        list_ans = short_ans.split('|')
        
        wiki_titles = []
        for ans in list_ans:
            
            ans = ans.strip()
            if ans:
                print(f"Processing answer: {ans}")
                title = get_title_1(ans)
                if not title:
                    title = get_title_2(ans)
                if not title:
                    title = get_title_3(ans)
                if not title:
                    title = get_title_4(ans)
                    
                if title:
                    print(f"Found title: {title}\n")
                    wiki_titles.append(title)                    
                else:
                    print("No title found.\n")
            time.sleep(3)
        raw_data.at[i, 'title_wiki'] = '|'.join(wiki_titles) if wiki_titles else ''
    
    output_file = r"data\bioasq_12b\output_B2_with_wiki.csv"
    raw_data.to_csv(output_file, index=False)
    print(f"File đã được lưu tại: {output_file}")
    
def diff_medhop():
    file = "data/differences_wiki_title.json"
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for ID, item in tqdm(data.items(), desc="Processing"):        
        query = item["Short Answer"]
        if not is_valid_query(query):
            print(f"ID: {ID}, Original Title: {query}, Invalid query format.")
            item["New Wiki Title"] = None
            continue
        
        title = get_title_1(query)
        if not title:
            title = get_title_2(query)
        if not title:
            title = get_title_3(query)
        if not title:
            title = get_title_4(query)

        if title:
            print(f"ID: {ID}, Original Title: {query}, Retrieved Title: {title}")
            item["New Wiki Title"] = title
        else:
            print(f"ID: {ID}, Original Title: {query}, No title found.")
            item["New Wiki Title"] = None
        time.sleep(15)
        
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def cwq(batch_size=50): 
    raw_data_file = "data/CWQ/cwq_ans_1000q.json"
    wiki_file = "data/CWQ/cwq_1000q_wiki_title.json"
    
    try:
        with open(wiki_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Resuming from existing file: {wiki_file}")
    except FileNotFoundError:
        with open(raw_data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Starting new processing from: {raw_data_file}")
    
    processed_count = 0
    total_items = len(data)

    for ID, item in tqdm(data.items(), desc = "processing", total=total_items):
        
        if "New Wiki Title" in item:
            processed_count += 1
            continue

        short_ans = item.get("short_ans") 
        
        if pd.isna(short_ans): 
            item["wiki_title"] = None
            continue
        
        list_ans = short_ans.split(',')
        
        wiki_titles = []
        for ans in list_ans:
            ans = ans.strip()
            
            if not is_valid_query(ans):
                print(f"ID: {ID}, Original Title: {ans}, Invalid query format.")
            else:
                print(f"Processing answer: {ans}")
                title = get_title_1(ans)
                if not title:
                    title = get_title_2(ans)
                if not title:
                    title = get_title_3(ans)
                if not title:
                    title = get_title_4(ans)
                
                if title:
                    print(f"Found title: {title}\n")
                    wiki_titles.append(title)
                else:
                    print("No title found.\n")
                time.sleep(5) 
                    
        item["New Wiki Title"] = wiki_titles
        
        processed_count += 1
        if processed_count % batch_size == 0:
            print(f"\n--- Batch save: Processed {processed_count}/{total_items} items. Saving to {wiki_file} ---")
            with open(wiki_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"\n--- Final save: Processing complete. Saving all data to {wiki_file} ---")
    with open(wiki_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("Done!")

def wikimultihop(batch_size=50):
    raw_data_file = r"data\wiki\dev\dev_compositional.csv"
    wiki_file = r"data\wiki\dev\dev_compositional_wiki.csv"

    try:
        raw_data = pd.read_csv(wiki_file)
        print(f"Resuming from existing file: {wiki_file}")
    except FileNotFoundError:
        raw_data = pd.read_csv(raw_data_file)
        raw_data['title_wiki'] = ''
        print(f"Starting new processing from: {raw_data_file}")

    processed_count = 0
    total_items = len(raw_data)

    for i, row in tqdm(raw_data.iterrows(), desc="processing", total=total_items):
        if row['title_wiki']:   # Nếu đã có thì bỏ qua (resume)
            processed_count += 1
            continue

        short_ans = row['Short Answer']
        if pd.isna(short_ans):
            raw_data.at[i, 'title_wiki'] = ''
            continue

        list_ans = short_ans.split('|')
        wiki_titles = []

        for ans in list_ans:
            ans = ans.strip()
            if not ans:
                continue

            if not is_valid_query(ans):
                print(f"Row {i}, Original Title: {ans}, Invalid query format.")
            else:
                print(f"Processing answer: {ans}")
                title = get_title_1(ans)
                if not title:
                    title = get_title_2(ans)
                if not title:
                    title = get_title_3(ans)
                if not title:
                    title = get_title_4(ans)

                if title:
                    print(f"Found title: {title}\n")
                    wiki_titles.append(title)
                else:
                    print("No title found.\n")
                time.sleep(3)

        raw_data.at[i, 'title_wiki'] = '|'.join(wiki_titles) if wiki_titles else ''
        processed_count += 1

        # batch save
        if processed_count % batch_size == 0:
            print(f"\n--- Batch save: Processed {processed_count}/{total_items} items. Saving to {wiki_file} ---")
            raw_data.to_csv(wiki_file, index=False)

    # final save
    print(f"\n--- Final save: Processing complete. Saving all data to {wiki_file} ---")
    raw_data.to_csv(wiki_file, index=False)
    print("Done!")

    
if __name__ == "__main__":
    # bioasq()
    # cwq(batch_size=50)
    wikimultihop()
    
