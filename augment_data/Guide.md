# Hướng Dẫn Sử Dụng - Data Augmentation Pipeline

## Mục Lục
1. [Giới Thiệu](#giới-thiệu)
2. [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
3. [Cài Đặt](#cài-đặt)
4. [Cấu Hình](#cấu-hình)
5. [Tạo Dữ Liệu](#tạo-dữ-liệu)
6. [Hợp Nhất Dữ Liệu](#hợp-nhất-dữ-liệu)
7. [Phân Tích Ablation](#phân-tích-ablation)

---

## Giới Thiệu

Pipeline này được thiết kế để tạo dữ liệu tổng hợp cho bài toán phân loại câu hỏi y sinh học (Biomedical QA Classification). Hệ thống tạo ra 3 loại câu hỏi:

| Loại | Mô Tả | Label |
|------|-------|-------|
| **Direct (Simple/Hard)** | Câu hỏi một bước, trả lời trực tiếp | 0 |
| **Sequential** | Câu hỏi nhiều bước, cần suy luận qua các entity trung gian | 1 |
| **Parallel** | Câu hỏi so sánh hai nhánh | (tùy chọn) |

---

## Cấu Trúc Thư Mục

```
augment_data/
├── config.yml                 # Cấu hình chính
├── template_config.json       # Bật/tắt từng template
├── generate_samples.py        # Script tạo mẫu đơn lẻ
├── batch_generate.py          # Script tạo mẫu hàng loạt
├── merge_data.py              # Hợp nhất dữ liệu
├── create_ablation_table.py   # Tạo bảng ablation
├── synthetic_data_generator.py # Định nghĩa templates
├── mesh_entity_extractor.py   # Trích xuất entity từ MeSH
├── output/
│   ├── csv/                   # Các file CSV đầu ra
│   ├── generated/             # JSON files được tạo
│   └── cache/                 # Cache entity đã sử dụng
└── train_set_used.csv         # Tập train
└── val_set_used.csv           # Tập validation
```

---

## Cài Đặt

### Yêu Cầu
- Python 3.8+
- API key cho OpenAI/GPT-4o-mini

### Cài Đặt Dependencies
```bash
pip install pyyaml requests
```

### Cấu Hình API Key
Đảm bảo file `src/api_key.py` chứa:
```python
API_LIST_4O = ["your-api-key-1", "your-api-key-2"]
API_LIST_O3 = ["your-api-key-1", "your-api-key-2"]
```

---

## Cấu Hình

### File `config.yml`

```yaml
generation:
  question_type: sequential    # direct_simple, direct_hard, sequential, parallel
  template_pattern: "chromosome_gene_disease"  # Tên template cụ thể
  num_samples: 20              # Số mẫu cần tạo
  batch_size: 20               # Kích thước mỗi batch
  reset_version: true          # true = version mới, false = tiếp tục version cũ

llm:
  model: gpt-4o-mini
  temperature: 0.8
  max_tokens: 4096
```

### File `template_config.json`

Bật (1) hoặc tắt (0) từng template:

```json
{
  "direct_hard": {
    "autoimmune_multi_symptom": 1,
    "syndrome_gene_cause": 0,
    ...
  },
  "sequential": {
    "chromosome_gene_disease": 1,
    "gene_mutation_condition": 1,
    ...
  }
}
```

---

## Tạo Dữ Liệu

### Tạo Đơn Lẻ (1 template)

1. Chỉnh sửa `config.yml`:
```yaml
generation:
  question_type: sequential
  template_pattern: "chromosome_gene_disease"
  num_samples: 20
```

2. Chạy:
```bash
python generate_samples.py
```

3. Kết quả lưu tại: `output/generated/sequential_chromosome_gene_disease_vX_Y.json`

### Tạo Hàng Loạt (tất cả templates)

```bash
python batch_generate.py
```

Script sẽ:
- Quét các file đã tạo để tránh duplicate
- Tạo đủ số mẫu cho mỗi template
- Hiển thị tiến độ

---

## Hợp Nhất Dữ Liệu

### Hợp nhất JSON thành CSV

```bash
python merge_data.py
```

Kết quả:
- `output/csv/merged_dataset.csv` - Tất cả dữ liệu
- `output/csv/direct_simple.csv` - Chỉ direct simple
- `output/csv/direct_hard.csv` - Chỉ direct hard
- `output/csv/sequential.csv` - Chỉ sequential

### Định dạng CSV đầu ra

| Cột | Mô Tả |
|-----|-------|
| QIDX | ID câu hỏi |
| Question | Nội dung câu hỏi |
| label | 0 = Direct, 1 = Sequential |
| biokgbench | 0 = Synthetic, 1 = BioKGBench |
| template | Tên template |

---

## Phân Tích Ablation

### Tạo Bảng Ablation

```bash
python create_ablation_table.py
```

Đầu vào: `output/ablation_results.json`
Đầu ra: `output/ablation_performance_table.csv`

### Cấu trúc bảng

| Cột | Mô Tả |
|-----|-------|
| category | direct_hard, direct_simple, sequential |
| ablated_template | Template bị loại bỏ |
| template_structure | Cấu trúc đầy đủ của template |
| P_direct, P_seq | Precision cho class 0 và 1 |
| R_direct, R_seq | Recall cho class 0 và 1 |
| F1_direct, F1_seq | F1 cho class 0 và 1 |
| Macro_F1 | Macro F1 tổng hợp |

---

## Lưu Ý Quan Trọng

### Entity Management
- Hệ thống cache các entity đã sử dụng để tránh trùng lặp
- File cache: `output/cache/{type}_cache.json`
- Đặt `reset_version: true` để bắt đầu mới

### Rate Limiting
- Script tự động xoay vòng API keys
- Chờ 0.5s giữa các template để tránh rate limit

### Kiểm Tra Chất Lượng
- Xem file `output/generation.log` để debug
- Kiểm tra trường `reasoning` trong JSON để đánh giá logic

---

## Troubleshooting

| Vấn Đề | Giải Pháp |
|--------|-----------|
| `No API keys` | Kiểm tra `src/api_key.py` |
| `Template not found` | Kiểm tra tên trong `synthetic_data_generator.py` |
| `Rate limit exceeded` | Tăng thời gian chờ trong `batch_generate.py` |
| Câu hỏi không đúng logic | Kiểm tra prompt trong `generate_samples.py` |

---

## Liên Hệ

Nếu có vấn đề, vui lòng kiểm tra:
1. `output/generation.log` - Log chi tiết
2. JSON files trong `output/generated/` - Dữ liệu thô
3. Cache files trong `output/cache/` - Entity đã dùng
