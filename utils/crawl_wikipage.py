import os
import json
import requests
from bs4 import BeautifulSoup
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import wikipediaapi  
 
def crawl_wikipedia_tool_2(title, page_language, url, base_delay=1):
    delay = base_delay
    time.sleep(delay)
    
    print('[Crawl wiki - Tool 2: wikipediaapi]')
    
    crawl_output = {
        'title': title,
        'url': url,
        'language': page_language,
        'source': 'tool 2 - wikipediaapi',
        'content': [],
        'sections': {}
    }

    wiki = wikipediaapi.Wikipedia(
        user_agent="MyResearchBot/1.0 (contact: vuthiminhthu04@gmail.com)",
        language=page_language,
        timeout=30 
    )

    page = wiki.page(title)

    # full page text
    crawl_output['content'] = [page.text]

    all_sec = {"Introduction": {'content': [page.summary]}}
    for s in page.sections: 
        sec_title = s.title 
        sec_content = s.text #string 
        all_sec[sec_title] = {'content': [sec_content]}
    
    crawl_output['sections'] = all_sec
    
    return crawl_output
    
    
def crawl_wikipedia_page(wiki_url, max_retries=3, base_delay=1):
    """
    Crawl a Wikipedia page and extract its content with proper rate limiting
    """
    print('[Crawl wiki - Tool 1: requests]')
    # Set up session with retry strategy
    session = requests.Session()
    
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # Realistic browser headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1'
    }
    
    # Add a delay with some randomness to appear more human-like
    delay = base_delay
    time.sleep(delay)
    
    try:
        # Make the request
        response = session.get(wiki_url, headers=headers, timeout=60)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title_element = soup.find('h1', {'class': 'firstHeading'})
        if not title_element:
            print(f"Warning: Could not find title for {wiki_url}")
            title = "Unknown Title"
        else:
            title = title_element.text.strip()
        
        # Extract main content
        content_div = soup.find('div', {'class': 'mw-parser-output'})
        if not content_div:
            content_div = soup.find('div', {'class': 'mw-page-container'})
        
        if not content_div:
            print(f"Warning: Could not find main content for {title}: {wiki_url}")
            return None
        
        # Remove unwanted elements
        for element_to_remove in content_div.select('table, .vector-toc-contents, figure, .thumb, .thumbinner, .floatright, .floatleft, .mw-file-description'):
            element_to_remove.decompose()
        
        # Process content by sections
        sections = {}
        current_heading = "Introduction"
        current_content = []
        
        # Get all headings, paragraphs, and lists
        all_elements = content_div.find_all(['h2', 'p', 'ul', 'ol'])
        
        for element in all_elements:
            # Skip if element is inside a table or other unwanted container
            if element.find_parent(['table', 'figure']) or element.find_parent('div', class_=['thumb', 'thumbinner', 'floatright', 'floatleft', 'mw-file-description']):
                continue
            
            # Process section headers    
            if element.name == 'h2':
                # Save previous section if it has content
                if current_content:
                    sections[current_heading] = {'content': current_content}
                
                # Start new section
                heading_text = element.get_text().strip()
                # Remove [edit] links from heading
                heading_text = heading_text.split('[')[0].strip()
    
                current_heading = heading_text
                current_content = []
            
            # Process paragraphs
            elif element.name == 'p':
                text = element.get_text().strip()
                if text and len(text) > 10 and len(text.split()) > 3:  # Filter out very short paragraphs
                    current_content.append(text)
            
            # Process lists
            elif element.name in ['ul', 'ol']:
                for li in element.find_all('li', recursive=False):
                    text = li.get_text().strip()
                    if text and len(text) > 5 and len(text.split()) > 3:
                        li_text = "<info> " + text
                        current_content.append(li_text)
        
        # Add the last section
        if current_content:
            sections[current_heading] = {'content': current_content}
        
        # Format content for output
        content = []
        for heading, section_data in sections.items():
            content.append(f"=== {heading} ===")
            content.extend(section_data['content'])
            content.append("")  # Add empty line between sections
        
        return {
            'title': title,
            'url': wiki_url,
            'source': 'tool 1 - request',
            'content': content,
            'sections': sections  # Also return structured sections
        }
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error crawling {wiki_url}: {e}")
        if hasattr(e, 'response') and e.response.status_code == 429:
            print(f"Rate limit exceeded. Consider increasing delay time.")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error crawling {wiki_url}: {e}")
        return None
    except requests.exceptions.Timeout as e:
        print(f"Timeout Error crawling {wiki_url}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request Error crawling {wiki_url}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error crawling {wiki_url}: {e}")
        return None
        
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())
      
import re
import json
from urllib.parse import urlparse
# import urllib.parse

def normalize_title_multilang(title):
    # Loại bỏ các hậu tố Wikipedia ở nhiều ngôn ngữ
    patterns = [
        r" - wikipedia.*$",  # tiếng Anh
        r"- 维基百科.*$",     # tiếng Trung
        r"- Википедия.*$",   # tiếng Nga
        r"- Wikipédia.*$",   # tiếng Pháp/Bồ Đào Nha
        r"- Вікіпедія.*$",   # tiếng Ukraina
        r"- Вікіпедыя.*$",   # tiếng Belarus
        r"- ויקיפדיה.*$",    # tiếng Hebrew
        r"- ויקיפדיה.*$",    # tiếng Hebrew
        r"- ويكيبيديا.*$",    # tiếng Ả Rập
        r"- ויקיפדיה.*$",    # tiếng Hebrew
        r"- Wikipedia.*$",   # fallback
    ]
    for pat in patterns:
        title = re.sub(pat, "", title, flags=re.IGNORECASE).strip()
    return title

if __name__ == "__main__":
    # wiki_url = "https://en.wikipedia.org/wiki/13q_deletion_syndrome"
    # wiki_url =  "https://en.wikipedia.org/wiki/Race_(human_categorization)"
    # wiki_url = "https://en.wikipedia.org/wiki/Puberty"
    # # wiki_url = "https://en.wikipedia.org/wiki/STAC3"
    # wiki_file_crawl = "wiki_crawl.json"
    # content = crawl_wikipedia_page(wiki_url)
    # print(content["sections"].keys())
    # save_json(content, wiki_file_crawl)
    
    
    with open('data/cache/IR_20_search_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_data = {}
    for link, item in data.items():
        
        norm_link = normalize_query(link)
        netloc = urlparse(norm_link).netloc
        lang = netloc.split('.')[0] if netloc.endswith('wikipedia.org') else ''
        item['language'] = lang
        new_data[norm_link] = item

    with open('data/cache/IR_20_search_results.json', 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)