# UETQuintet - Multi-hop Question Answering System

UETQuintet là một hệ thống trả lời câu hỏi đa bước (multi-hop QA) sử dụng các kỹ thuật phân tách câu hỏi, trích xuất thực thể và tạo đáp án tự động. Hệ thống hỗ trợ nhiều dataset khác nhau bao gồm BioASQ, MedHop, ComplexWebQuestions (CWQ), và WikipediaQA.

## 🔧 Yêu cầu hệ thống

- Python 3.9
- API key cho GPT models

## 📦 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/quocanuetcs/UETQuintet.git
cd UETQuintet
```

### 2. Cài đặt dependencies

```bash
pip install -r requirments.txt
```

### 3. Cài đặt SpaCy models

```bash
# Standard English model
python -m spacy download en_core_web_lg

# Biomedical NER model (optional, chỉ cần cho biomedical datasets)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
```

### 4. Cấu hình API key

Tạo file `src/api_key.py` với nội dung:

```python
# LLM API key
API_LIST_4O = ["your-openai-api-key-here"]
API_LIST_O3 = ["your-openai-api-key-here"]

# Google Search API (optional)
API_KEY_GG = "your-google-api-key-here"
```

**Lưu ý**: File `src/api_key.py` đã được thêm vào `.gitignore` để bảo mật.

## 🚀 Cách chạy

### Cấu hình cơ bản

Chỉnh sửa file `main.py` để cấu hình các tham số:

```python
# Chọn cấu hình
run = runs[5]        # R11 (index 5)
threads = [0, 2]     # Xử lý từ sample 0 đến 2
dataset = datasets[3] # wiki (index 3)
batch = batchs[0]    # b1 (index 0)
important = True     # Chỉ dùng cho medhop
```

### Các dataset có sẵn

1. **BioASQ** (`datasets[0]`): y sinh 
2. **MedHop** (`datasets[1]`): Multi-hop medical reasoning
3. **CWQ** (`datasets[2]`): Complex Web Questions
4. **Wiki** (`datasets[3]`): 2Wikimultihop

### Các run configuration

- **R1**: Chỉ sử dụng Google Search
- **R4**: Google Search + Wikipedia + Crawl Wikipedia
- **R5**: Tương tự R4, tất cả là non-seq
- **R6**: Wikipedia + Crawl Wikipedia (không Google)
- **R10**: Không sử dụng external retrieval
- **R11**: Key entity search + Context từ 2Wiki
- **R12**: Key entity search + Context từ 2Wiki + Wikipedia 

### Chạy chương trình

```bash
python main.py
```

### Chạy với các dataset cụ thể

#### 1. MedHop Dataset
```python
# Trong main.py
dataset = datasets[1]  # medhop
batch = batchs[0]      # b1 (unconfused samples)
important = False      # Sử dụng toàn bộ dataset
threads = [0, 20]      # Xử lý 20 samples
```

#### 2. BioASQ Dataset
```python
# Trong main.py
dataset = datasets[0]  # bioasq
batch = batchs[0]      # b1 (bioasq_12b1)
threads = [0, 5]       # Xử lý 5 câu hỏi đầu tiên
```

#### 3. CWQ Dataset
```python
# Trong main.py
dataset = datasets[2]  # cwq
batch = batchs[0]      # b1
threads = [0, 10]      # Xử lý 10 câu hỏi đầu tiên
```

#### 4. Wiki Dataset
```python
# Trong main.py
dataset = datasets[3]  # wiki
batch = batchs[0]      # b1
threads = [0, 10]      # Xử lý 10 câu hỏi đầu tiên
```

## 📁 Cấu trúc dữ liệu

### Output structure

Kết quả được lưu trong các thư mục:
- `wiki_preprocessing/`: Kết quả phân tách câu hỏi
- `wiki_results/`: Kết quả cuối cùng
- Cấu trúc: `{dataset}_{run}_step_{step}_{type}_{sample_id}.json`

## ⚙️ Cấu hình chi tiết

### File config.json

File này chứa cấu hình cho từng dataset và run:

```json
{
  "wiki": {
    "R11": {
      "dataset": "wiki_",
      "get_gg": "False",
      "get_wiki": "False", 
      "get_crawl_wiki": "False",
      "search_key_entity": "True",
      "get_context_2wiki": "True"
    }
  }
}
```

### Các tham số quan trọng

- `get_gg`: Sử dụng Google Search
- `get_wiki`: Sử dụng Wikipedia API
- `get_crawl_wiki`: Crawl Wikipedia pages
- `search_key_entity`: Tìm kiếm theo key entities
- `get_context_2wiki`: Sử dụng context từ 2Wiki

## 🔍 Pipeline xử lý

1. **Question Decomposition**: Phân tách câu hỏi phức tạp thành các sub-questions
2. **Entity Extraction**: Trích xuất các thực thể quan trọng
3. **Information Retrieval**: Thu thập thông tin từ các nguồn
4. **Answer Generation**: Tạo đáp án cho từng sub-question
5. **Answer Combination**: Kết hợp các đáp án thành câu trả lời cuối cùng
