import requests
import requests
from bs4 import BeautifulSoup

def get_wiki_abstract(url): 
    """
    Lấy phần tóm tắt (abstract) của một bài viết từ Wikipedia tiếng Anh.

    Phần tóm tắt được định nghĩa là các đoạn văn bản (<p>)
    trước khi đến mục lục hoặc tiêu đề chính đầu tiên (<h2>).
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        content_div = soup.find('div', class_='mw-parser-output')
        if not content_div:
            return None

        abstract_paragraphs = []
        for element in content_div.children:
            if element.name == 'h2' or (element.name == 'div' and element.get('id') == 'toc'):
                break
            
            if element.name == 'p':
                # Bỏ qua các đoạn trống hoặc chỉ chứa toạ độ (thường có class đặc biệt)
                if 'geo-nondefault' in element.get('class', []):
                    continue
                abstract_paragraphs.append(element.get_text(strip=True))

        if not abstract_paragraphs:
            print(url)
            print("❌ Lỗi: Không tìm thấy đoạn văn nào trong phần tóm tắt.")
            return None

        return '\n\n'.join(abstract_paragraphs)

    except requests.exceptions.RequestException as req_err:
        return None

def crawl_abstract_wiki(title, article_url, lang="en"):
    # Chuẩn hóa tiêu đề để dùng cho URL
    formatted_title = title.replace(' ', '_')

    # 1. Thử dùng API chính thức 
    api_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{formatted_title}"
    try:
        response = requests.get(api_url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            extract = data.get("extract")
            if extract:
                return extract
    except requests.exceptions.RequestException as e:
        print(title)
        print(article_url)
        print(f"Lỗi khi gọi API: {e}. Chuyển sang phương pháp scraping...")

    return get_wiki_abstract(article_url)

# def crawl_abstract_wiki(title, url, lang="en"):
#     title = title.replace(' ', '_')
#     url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.json()
#         return data.get("extract")  # Trả về abstract
#     else:
#         return get_wiki_abstract(url)

if __name__ == "__main__":
    link = "https://en.wikipedia.org/wiki/Move_(1970_film)"
    title = "Move (1970 film)"
    # abstract = get_wiki_abstract(link)
    abstract = crawl_abstract_wiki(title, link, lang="en")
    print("\n--- Tóm tắt cho:", link, "---\n")
    print(abstract)

    print("\n" + "="*50 + "\n")

   