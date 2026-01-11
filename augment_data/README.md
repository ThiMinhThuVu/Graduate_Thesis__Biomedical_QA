# Synthetic Biomedical QA Generation Pipeline

Generate multi-hop biomedical questions using MeSH entities with controlled entity usage.

## Quick Start

```bash
# Run generation (uses config.yml)
python synthetic_qa/generate_samples.py
```

Output: `synthetic_qa/output/generated_samples.json`

---

## Pipeline Overview

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  MeSH Data (22K)    │────▶│  Entity Extractor │────▶│  LLM Generator  │
│  shallow_query_     │     │  (randomized,     │     │  (batch mode,   │
│  result_2.json      │     │   no repeats)     │     │   per type)     │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
                                                              │
                                                              ▼
                                                     ┌─────────────────┐
                                                     │  Output JSON    │
                                                     │  (15 questions) │
                                                     └─────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Batch Generation** | 1 API call per question type (3 calls total) |
| **No Entity Repeats** | Each MeSH entity used only once |
| **Randomized Selection** | Entities shuffled each run |
| **3 Question Types** | Direct, Sequential, Parallel |

---

## Files

| File | Purpose |
|------|---------|
| `generate_samples.py` | Main script - run this |
| `mesh_entity_extractor.py` | Extracts entities from MeSH data |
| `data_generation_prompt.py` | Prompt templates and definitions |
| `synthetic_data_generator.py` | Template structures |
| `config.yml` | Configuration settings |

---

## Configuration (config.yml)

```yaml
# LLM settings
llm:
  model: "gpt-4o-mini"
  temperature: 0.7

# Generation settings
generation:
  num_per_type: 5              # Questions per type (direct/sequential/parallel)
  question_types: "all"        # "all" or ["direct", "sequential"]
  output: "output/generated_samples.json"

# Entity source: "mesh" or "manual"
entity_source: "mesh"

# MeSH entity counts (per batch, each used only once)
# Total available: 22,216 entities
mesh_entities:
  entity_counts:
    diseases: 500              # from 4,911 available
    chemicals_drugs: 500       # from 10,277 available
    anatomy: 200               # from 1,857 available
    phenomena_processes: 200   # from 2,358 available
    techniques_equipment: 100  # from 3,018 available
    genes: 100                 # filtered from D
    proteins: 100              # filtered from D
    symptoms: 100              # filtered from C
    drugs: 100                 # filtered from D
```

---

## Question Types

### Direct
Single-step reasoning about one entity.
```
Q: What enzyme is deficient in phenylketonuria?
A: Phenylalanine hydroxylase
```

### Sequential  
Multi-hop chain: A → B → Answer
```
Q: Which gene is mutated in the disease treated with Imiglucerase?
A: GBA gene
Chain: Imiglucerase → Gaucher disease → GBA
```

### Parallel
Compare/combine independent branches.
```
Q: Are Duchenne and Becker muscular dystrophy caused by the same gene?
A: Yes, DMD gene
Branch 1: Duchenne → DMD
Branch 2: Becker → DMD
```

---

## MeSH Categories

Entities extracted from `shallow_query_result_2.json` (22,216 terms):

| Code | Category | Count |
|------|----------|-------|
| C | Diseases | 4,911 |
| D | Chemicals and Drugs | 10,277 |
| A | Anatomy | 1,857 |
| G | Phenomena and Processes | 2,358 |
| E | Techniques and Equipment | 3,018 |
| B | Organisms | 29 |

---

## Output Format

```json
{
  "metadata": {
    "generated_at": "2026-01-10T12:30:00",
    "num_per_type": 5,
    "types": ["direct", "sequential", "parallel"]
  },
  "entity_pools": {
    "direct": {"diseases": [...], "genes": [...], ...},
    "sequential": {"diseases": [...], ...},
    "parallel": {"diseases": [...], ...}
  },
  "samples": [
    {
      "id": "direct_1",
      "question_type": "direct",
      "status": "success",
      "generated": {
        "question": "Which disease is characterized by...",
        "short_answer": "Gaucher disease",
        "reasoning": "Step 1: ... Step 2: ...",
        "topic": "Pathophysiology",
        "entities_used": ["GBA", "Gaucher disease"]
      }
    }
  ],
  "stats": {
    "total": 15,
    "success": 15,
    "unique_entities": 35,
    "duplicates": {}
  }
}
```

---

## Usage Examples

### Generate with default settings
```bash
python synthetic_qa/generate_samples.py
```

### Modify config for more samples
```yaml
# In config.yml
generation:
  num_per_type: 10  # 10 per type = 30 total
```

### Generate specific types only
```yaml
generation:
  question_types: ["sequential", "parallel"]  # Skip direct
```

### Use manual entities instead of MeSH
```yaml
entity_source: "manual"

entities:
  diseases:
    - "Gaucher disease"
    - "Fabry disease"
  genes:
    - "GBA"
    - "GLA"
```

---

## API Requirements

Requires API keys in `src/api_key.py`:
```python
API_LIST_4O = ["key1", "key2", ...]  # for gpt-4o-mini
API_LIST_O3 = ["key1", "key2", ...]  # for o3-mini
```

---

## Entity Extraction API

```python
from synthetic_qa import mesh_entity_extractor as mesh

# Load data (call once)
mesh.load_mesh_data(seed=42)

# Get unused entities
diseases = mesh.get_entities("C", count=10)
drugs = mesh.get_entities("D", count=10)

# Get full pool for generation
pool = mesh.get_entity_pool({"default_entity_count": 10})

# Check usage
stats = mesh.get_stats()
print(f"Used: {stats['used']}, Remaining: {stats['remaining']}")

# Reset for new run
mesh.reset()
```

---

## Tips

1. **Increase diversity**: Set higher `num_per_type` and entity counts
2. **Different runs**: Each run uses random seed based on timestamp
3. **Check duplicates**: Output stats show if any entity was repeated
4. **Quality control**: Review `reasoning` field for factual accuracy
