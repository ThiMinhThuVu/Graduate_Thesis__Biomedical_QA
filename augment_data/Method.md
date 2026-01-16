# Phương Pháp Luận - Tạo Dữ Liệu Tổng Hợp cho Phân Loại Câu Hỏi Y Sinh

## Mục Lục
1. [Tổng Quan](#tổng-quan)
2. [Phân Loại Câu Hỏi](#phân-loại-câu-hỏi)
3. [Thiết Kế Template](#thiết-kế-template)
4. [Quy Trình Tạo Dữ Liệu](#quy-trình-tạo-dữ-liệu)
5. [Đánh Giá và Ablation](#đánh-giá-và-ablation)
6. [Kết Quả và Phân Tích](#kết-quả-và-phân-tích)

---

## Tổng Quan

### Mục Tiêu
Xây dựng pipeline tạo dữ liệu tổng hợp để huấn luyện mô hình phân loại câu hỏi y sinh học thành hai loại:
- **Direct (Label 0)**: Câu hỏi có thể trả lời trực tiếp trong một bước
- **Sequential (Label 1)**: Câu hỏi cần suy luận nhiều bước qua các entity trung gian

### Động Lực
Dữ liệu y sinh học thực tế thường:
- Khan hiếm và khó thu thập
- Không cân bằng giữa các loại câu hỏi
- Thiếu đa dạng về cấu trúc ngữ pháp

Pipeline này giải quyết các vấn đề trên bằng cách tạo dữ liệu tổng hợp đa dạng, có kiểm soát.

---

## Phân Loại Câu Hỏi

### 1. Direct Simple (Label 0)
**Đặc điểm:**
- Một entity được cho trước
- Câu trả lời có thể tìm được trực tiếp
- Cấu trúc đơn giản

**Ví dụ:**
```
Q: Is tuberculosis a notifiable disease?
A: Yes

Chain: [DISEASE] → [PROPERTY]
```

**Templates (16):**
- `which_specialty`: Chuyên khoa điều trị
- `is_notifiable`: Bệnh bắt buộc báo cáo
- `can_lead_to`: Biến chứng có thể xảy ra
- `on_chromosome`: Vị trí gene trên chromosome
- ...

### 2. Direct Hard (Label 0) - Hard Negatives
**Đặc điểm:**
- Nhiều entity được cho trước (TRÔNG GIỐNG multi-hop)
- Nhưng vẫn là câu hỏi một bước
- Dùng làm "hard negatives" cho training

**Ví dụ:**
```
Q: Which gene mutation causes a form of muscular dystrophy that is also classified as a myopathy?
A: DMD

Chain: [DISEASE_TYPE_A] + [DISEASE_TYPE_B] → [GENE] (single lookup)
```

**Lý do thiết kế:**
- Mô hình cần học phân biệt câu hỏi "phức tạp về ngữ pháp" vs "phức tạp về logic"
- Direct Hard giúp mô hình không bị đánh lừa bởi độ dài/phức tạp bề ngoài

**Templates (12):**
- `autoimmune_multi_symptom`: Bệnh tự miễn với nhiều triệu chứng
- `syndrome_multi_feature`: Hội chứng với nhiều đặc điểm
- `gene_dual_phenotype`: Gene gây nhiều kiểu hình
- ...

### 3. Sequential (Label 1)
**Đặc điểm:**
- CHỈ entity khởi đầu được cho
- Các entity trung gian phải được suy luận/khám phá
- Câu trả lời ở cuối chuỗi suy luận

**Ví dụ:**
```
Q: Which chromosome contains the gene that causes Marfan syndrome?
A: Chromosome 15

Chain: [DISEASE: Marfan] → [GENE?: FBN1] → [CHROMOSOME?: 15]
       (given)            (implicit)       (answer)
```

**Ký hiệu trong template:**
- `[ENTITY]` = Entity được cho trong câu hỏi
- `[ENTITY?]` = Entity ẩn, cần khám phá

**Templates (31, bao gồm cũ và mới):**

| Template | Coverage | Chain |
|----------|----------|-------|
| `chromosome_gene_disease` | 47% | Disease → Gene? → Chromosome? |
| `gene_mutation_condition` | 11% | Symptom → Condition? → Gene? |
| `inheritance_pattern_condition` | - | Gene → Condition? → Inheritance? |
| `pathogen_disease_symptom` | 5% | Symptom → Disease? → Pathogen? |
| ... | | |

---

## Thiết Kế Template

### Nguyên Tắc

1. **Cấu trúc rõ ràng**: Mỗi template có structure và chain xác định
2. **Entity đánh dấu**: `?` cho entity implicit
3. **Đa dạng wh-word**: Which, What, How, Is, Can, Do
4. **Phân bố chủ đề**: Pathophysiology, Treatment, Diagnosis, Prevention

### Cấu Trúc Template

```python
{
    "pattern": "chromosome_gene_disease",
    "structure": "Which [CHROMOSOME?] contains the [GENE?] that causes [DISEASE]?",
    "chain": "[DISEASE] → [GENE?] → [CHROMOSOME?]",
    "wh_word": "Which",
    "topic": "Pathophysiology",
    "example": "Which chromosome contains the gene that causes Marfan syndrome?"
}
```

### Phân Tích MedHop Missed Sequential

Từ 75 câu hỏi sequential bị phân loại sai, chúng tôi phân tích và tạo templates mới:

| Template Mới | % Coverage | Nguồn |
|--------------|-----------|-------|
| `chromosome_gene_disease` | 47% | Missed analysis |
| `inheritance_pattern_condition` | - | Missed analysis |
| `gene_property_count` | - | Missed analysis |
| `gene_family_member` | - | Missed analysis |
| `disease_symptom_combo` | - | Missed analysis |

---

## Quy Trình Tạo Dữ Liệu

### 1. Trích Xuất Entity
```
MeSH Database → mesh_entity_extractor.py → Entity Pool (22,216 entities)
```

**Loại entity:**
- Diseases: Bệnh, hội chứng
- Genes: Gene, protein
- Chemicals: Thuốc, hợp chất
- Anatomy: Cơ quan, mô
- Organisms: Vi sinh vật

### 2. Xây Dựng Prompt

**System Prompt:**
```
You are a biomedical QA expert. Generate {num} natural-sounding {qtype} questions following the template pattern.

Rules:
- Questions should sound natural, like real biomedical exam questions
- Follow the template's reasoning structure, but vary the wording
- Use the provided entities as starting points
```

**User Prompt:**
```
Generate {num} {qtype} biomedical questions using this template:

TEMPLATE: {template_pattern}
{template_structure}

ENTITIES TO USE:
- Entity 1
- Entity 2
...

RULE: {type_specific_rule}

OUTPUT FORMAT (JSON array):
[{"question": "...", "short_answer": "...", "reasoning": "...", "topic": "...", "pattern": "...", "entities_used": [...]}]
```

### 3. Gọi LLM và Parse

```
Prompt → GPT-4o-mini → JSON Response → Validation → Save
```

**Validation:**
- Kiểm tra JSON format
- Kiểm tra các trường bắt buộc
- Loại bỏ duplicate

### 4. Cache Management

```python
{
    "version": "v5_0",
    "used_entity_ids": ["D001234", "D005678", ...],
    "template_usage": {
        "chromosome_gene_disease": 20,
        ...
    }
}
```

---

## Đánh Giá và Ablation

### Ablation Study Design

**Mục tiêu:** Đánh giá đóng góp của từng template

**Phương pháp:**
1. Train baseline với tất cả templates
2. Với mỗi template T:
   - Loại bỏ T khỏi training set
   - Train lại mô hình
   - So sánh metrics với baseline

**Metrics:**
| Metric | Ý Nghĩa |
|--------|---------|
| P_direct | Precision class 0 |
| P_seq | Precision class 1 |
| R_direct | Recall class 0 |
| R_seq | Recall class 1 |
| F1_direct | F1 class 0 |
| F1_seq | F1 class 1 |
| Macro_F1 | Trung bình F1 |

### Phân Tích Kết Quả

**Template quan trọng nhất** (loại bỏ làm giảm Macro_F1 nhiều nhất):
- `deficiency_associated_disorder`: Baseline 0.7525 → 0.6849 (-6.76%)
- `cell_role_pathogenesis`: Baseline 0.7525 → 0.7056 (-4.69%)
- `disorder_secondary_effect`: Baseline 0.7525 → 0.6862 (-6.63%)

**Template ít quan trọng** (loại bỏ không ảnh hưởng/cải thiện):
- `autoimmune_multi_symptom`: 0.7525 → 0.782 (+2.95%)
- `treatment_dual_role`: 0.7525 → 0.7797 (+2.72%)

---

## Kết Quả và Phân Tích

### Thống Kê Dữ Liệu Cuối

| Tập | Tổng | Direct (0) | Sequential (1) |
|-----|------|------------|----------------|
| Train | 967 | 595 (61.5%) | 372 (38.5%) |
| Validation | 171 | 105 (61.4%) | 66 (38.6%) |

### Phân Bố Template

**Direct Simple:** 16 templates, ~170 samples
**Direct Hard:** 12 templates, ~425 samples  
**Sequential (New):** 10 templates, ~200 samples
**Sequential (Orig):** 21 templates, ~170 samples

### Lessons Learned

1. **Hard Negatives quan trọng**: Direct Hard cải thiện khả năng phân biệt
2. **Template đa dạng**: Nhiều template → mô hình robust hơn
3. **Phân tích lỗi**: Missed sequential → templates mới
4. **Cân bằng**: Giữ tỷ lệ ~60/40 giữa Direct/Sequential

---

## Hướng Phát Triển

1. **Thêm templates mới** từ phân tích lỗi liên tục
2. **Improve prompt** để tạo câu hỏi tự nhiên hơn
3. **Multi-language** mở rộng sang tiếng Việt
4. **Active learning** chọn samples khó để annotate

---

## Tài Liệu Tham Khảo

- MeSH Database: https://www.ncbi.nlm.nih.gov/mesh
- BioKGBench Dataset
- MedHopQA Dataset
- GPT-4o-mini API Documentation
