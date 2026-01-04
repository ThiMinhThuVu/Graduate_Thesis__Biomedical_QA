# Project Architecture Diagram

Below is a high-level architecture diagram and description for the current project structure:

## Detailed Block Diagram

```
┌──────────────┐
│   INPUT      │
│──────────────│
│ - data/      │
│   (raw_data) │
│ - config/    │
│(config.yaml) │
└──────┬───────┘
	│
	▼
┌──────────────────────────────┐
│ main_hop2_wikidata_dev.py    │
└────────────┬────────────────┘
	      │
	      ▼
┌────────────────────────────────────────────────────────────┐
│ 1. Classification (classify_question)                     │
│    - classification/                                      │
└────────────┬───────────────────────────────────────────────┘
	      │
	      ▼
┌────────────────────────────────────────────────────────────┐
│ 2. Decomposition (decompose_question_parallel/sequential)  │
│    - src/decomposition.py                                  │
└────────────┬───────────────────────────────────────────────┘
	      │
	      ▼
┌────────────────────────────────────────────────────────────┐
│ 3. Retrieval (retrieve_evidence)                           │
│    - src/retrieval.py                                      │
└────────────┬───────────────────────────────────────────────┘
	      │
	      ▼
┌────────────────────────────────────────────────────────────┐
│ 4. NER (extract_entities_from_evidence)                    │
│    - src/extract_ner.py                                    │
└────────────┬───────────────────────────────────────────────┘
	      │
	      ▼
┌────────────────────────────────────────────────────────────┐
│ 5. apply_answer_normalization_rules                        │
│                                                            │
└────────────┬───────────────────────────────────────────────┘
	      │
	      ▼
┌────────────────────────────────────────────────────────────┐
│ 6. Answer Generation (generate_answer, etc.)               │
│    - src/generate_answer.py                                │
└────────────┬───────────────────────────────────────────────┘
	      │
	      ▼
┌────────────────────────────────────────────────────────────┐
│ 6. Normalization (normalize_answer_with_wikipedia, etc.)   │
│    - utils/norm_ans_with_wiki.py                           │
└────────────┬───────────────────────────────────────────────┘
	      │
	      ▼
┌────────────────────────────────────────────────────────────┐
│ 7. Multiple Choice                                         │
│    gen_ans_multiple_choice                                 │
└────────────┬───────────────────────────────────────────────┘
	      │
	      ▼
┌────────────────────────────────────────────────────────────┐
│ 8. Combine                                                 │
│    generate_combine_parallel_answers                       │
└────────────┬───────────────────────────────────────────────┘
	      │
	      ▼
┌──────────────────────────────┐
│   OUTPUT                     │
│──────────────┬───────────────│
│ - medhop_results/            │
│ - evaluate/                  │
└──────────────────────────────┘
```

### Flow Description
- **Input:** Dữ liệu (JSON, CSV) và cấu hình được nạp vào.
- **Classification:** Phân loại câu hỏi để xác định hướng xử lý.
- **Decomposition:** Phân rã câu hỏi phức tạp thành các sub-question.
- **Retrieval:** Truy xuất bằng chứng từ dữ liệu.
- **NER:** Trích xuất thực thể liên quan từ bằng chứng.
- **Normalization Rules:** Rules chuẩn hóa đáp án.
- **Answer Generation:** Sinh đáp án dựa trên bằng chứng và luật.
- **Normalization:** Chuẩn hóa đáp án nếu cần.
- **Multiple Choice/Combine:** Tổng hợp đáp án cuối cùng hoặc chọn đáp án đúng nhất.
- **Output:** Lưu kết quả vào medhop_results/ và evaluate/.

## Description
- **main_hop2_wikidata_dev.py**: Main entry point for the project.
- **classification/**: Contains classification scripts and models.
- **config/**: Configuration files.
- **data/**: Datasets, caches, and raw data.
- **evaluate/**: Evaluation scripts.
- **medhop_preprocessing/**: Preprocessing steps for MedHop data.
- **medhop_results/**: Stores results and evidence.
- **src/**: Source code for core logic and utilities.
- **utils/**: Utility scripts for data processing and crawling.

> This diagram provides a high-level overview. For more details, see the README or explore each folder.
