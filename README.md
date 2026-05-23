# UETQuintet

UETQuintet là hệ thống trả lời câu hỏi đa bước, gồm tách câu hỏi, truy xuất thông tin và tạo đáp án cuối.

## Cần chuẩn bị

- Python 3.9+
- API key cho model LLM
- SpaCy model `en_core_web_lg`
- Với MedHop, cài thêm các biomedical NER model

## Cài đặt

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## Cấu hình

Tạo `src/api_key.py` và điền API key cần dùng. Sau đó chỉnh `config/config.yaml` để chọn dataset và chế độ chạy.

## Chạy

Chạy script phù hợp trong thư mục gốc, ví dụ:

```bash
python main_final_1000q.py
```

## Kết quả

Kết quả và cache được ghi trong các thư mục kết quả tương ứng của từng chế độ/dataset trong repo.
