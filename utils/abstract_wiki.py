import requests
import re
from urllib.parse import quote
from bs4 import BeautifulSoup

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/110.0.0.0 Safari/537.36"
    )
}

def get_wiki_abstract(url):
    """
    Lấy phần tóm tắt (abstract) của một bài viết từ Wikipedia tiếng Anh.

    Phần tóm tắt được định nghĩa là các đoạn văn bản (<p>)
    trước khi đến mục lục hoặc tiêu đề chính đầu tiên (<h2>).
    """
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Lấy tiêu đề trang (title trong <h1> hoặc <title>)
        h1_tag = soup.find("h1")
        if h1_tag:
            page_title = h1_tag.get_text(strip=True)
        else:
            page_title = soup.title.string.strip() if soup.title else None

        content_div = soup.find("div", {"class": "mw-content-ltr mw-parser-output"})
        if not content_div:
            return None

        abstract_paragraphs = []
        for element in content_div.children:
            if element.name == 'h2' or (element.name == 'div' and element.get('id') == 'toc'):
                break
            
            if element.name == 'p':
                if 'geo-nondefault' in element.get('class', []):
                    continue
                abstract_paragraphs.append(element.get_text(separator=" ", strip=True))

        if not abstract_paragraphs:
            print(url)
            print("❌ Lỗi: Không tìm thấy đoạn văn nào trong phần tóm tắt.")
            return None

        abstract_text = ' '.join(abstract_paragraphs)
        # Loại bỏ các ký hiệu [1], [2], ...
        abstract_text = re.sub(r'\[\s*[0-9a-zA-Z]+\s*\]', '', abstract_text)
        # Xóa khoảng trắng thừa:
        abstract_text = re.sub(r'\s{2,}', ' ', abstract_text).strip()

        return abstract_text, page_title
    
    except requests.exceptions.RequestException as req_err:
        return None, None

def crawl_abstract_wiki(title, lang="en"):
    formatted_title = quote(title)
    print(formatted_title)

    # API này lấy snippet
    # api_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{formatted_title}"
    # try:
    #     response = requests.get(api_url, headers=headers, timeout=30)
    #     if response.status_code == 200:
    #         data = response.json()
    #         extract = data.get("extract")
    #         if extract:
    #             return extract
    # except requests.exceptions.RequestException as e:
    #     print(title)
    #     print(article_url)
    #     print(f"Lỗi khi gọi API: {e}. Chuyển sang phương pháp scraping...")

    # Có thể lấy được abstract
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,      # chỉ phần mở đầu
        "explaintext": True,  # chỉ text, không HTML
        "titles": formatted_title,
        "format": "json"
    }
    try:
        response = requests.get(api_url, headers=headers, params=params)
        if response == 200:
            data = response.json()
            page = next(iter(data["query"]["pages"].values()))
            return page.get("extract", "")       
    
    except requests.exceptions.RequestException as e:
        print(title)
        print(f"Lỗi khi gọi API: {e}. Chuyển sang phương pháp scraping...")

    article_url = f"https://en.wikipedia.org/wiki/{formatted_title}"
    abstract_text, page_title = get_wiki_abstract(article_url)
    return abstract_text, page_title

if __name__ == "__main__":
    test_entities = ["Prince Henri, Duke of Aumale"]

    for title in test_entities:
        print("\n" + "="*80)
        print(f"🔹 {title}")
        abstract, page_title = crawl_abstract_wiki(title)
        print(f"➡️ Abstract:\n{abstract or '[Không có]'}")
        print(f"➡️ Page Title: {page_title or '[Không có]'}")
   