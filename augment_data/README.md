# Augment Data - Synthetic Biomedical QA Generation

Generates synthetic biomedical multi-hop questions for training question type classifiers. Aligned with BioKGBench and BioHopR datasets.

## Quick Start

```bash
# 1. Edit config.yml to select question type
#    question_type: direct | sequential | parallel

# 2. Run generation
python generate_samples.py

# 3. Check output
ls output/generated/
```

## Directory Structure

```
augment_data/
├── config.yml                    # Configuration
├── generate_samples.py           # Main generation script
├── synthetic_data_generator.py   # Templates + versioned cache
├── mesh_entity_extractor.py      # MeSH entity extraction
├── data_generation_prompt.py     # Prompt templates & few-shot examples
│
├── output/
│   ├── cache/                    # Version cache per type
│   │   ├── direct_v1_0.json
│   │   ├── sequential_v1_0.json
│   │   └── parallel_v1_0.json
│   ├── generated/                # Generated samples
│   │   ├── direct_v1_0.json
│   │   ├── sequential_v1_0.json
│   │   └── parallel_v1_0.json
│   └── generation.log            # Activity log
│
└── shallow_query_result_2.json   # MeSH entity data
```

## Configuration (config.yml)

```yaml
# LLM settings
llm:
  model: "gpt-4o-mini"
  temperature: 0.7

# Generation Settings - ONE TYPE AT A TIME
generation:
  question_type: sequential   # direct | sequential | parallel
  num_samples: 500            # Samples to generate
  batch_size: 25              # Per API call
  reset_version: false        # false=v1_0→v1_1, true=v1→v2_0

entity_source: "mesh"
```

## Version System

Each question type has independent versioning:

| Action | Before | After | Entities |
|--------|--------|-------|----------|
| First run | none | `v1_0` | Fresh pool |
| Continue | `v1_0` | `v1_1` | Inherited exclusion |
| Continue | `v1_1` | `v1_2` | Inherited exclusion |
| Reset | `v1_x` | `v2_0` | Fresh pool |

### Entity Exclusion

Entities used in v1_0 are excluded in v1_1, v1_2, etc. to ensure distinct data:

```
v1_0: Uses entities A, B, C → Saved to cache
v1_1: Excludes [A,B,C] → Uses D, E, F
v2_0: Reset → Fresh pool, can use any entity
```

### Wording Diversity

The prompt encourages LLM to:
- **Follow template reasoning structure** (e.g., Gene → Disease → Treatment)
- **Rephrase wording creatively** - don't copy verbatim
- **Vary sentence structure**: "Which X...", "Name the...", "How does..."
- **Use synonyms**: "causes" → "leads to/results in/triggers"

### Template Balancing

Templates are automatically balanced for even distribution:
- Templates sorted by usage (least used first)
- Strong hints: `⚠️ MUST USE underused patterns`
- Target distribution: `📊 Target: ~N questions per pattern`
- Result: Variance reduced from 0-10 to 7-8 per pattern

### Files Created

- Cache: `output/cache/{type}_v{version}.json` (patterns + used_entity_ids)
- Output: `output/generated/{type}_v{version}.json` (samples + prompt_template)
- Log: `output/generation.log`

## Templates

### Direct (16 templates)
Single-hop questions with direct reasoning.

```
"Which medical specialty treats [DISEASE]?"
"How does [PROTEIN] contribute to [DISEASE]?"
"On which chromosome is [GENE] located?"
```

### Sequential (22 templates)
Multi-hop chains requiring step-by-step reasoning.

```
Chain: [GENE] → [DISEASE] → [DRUG]
"What drug treats the disease caused by [GENE] mutations?"

Chain: [SYMPTOM] → [DISEASE] → [GENE] → [CHROMOSOME]
"On which chromosome is the gene causing the disease with [SYMPTOM]?"
```

### Parallel (4 templates)
Comparison questions with independent branches.

```
Branches: [DISEASE_A] → gene, [DISEASE_B] → gene
"Is the gene mutated in [A] from the same family as in [B]?"
```

## CLI Commands

```bash
# Show template counts and versions
python synthetic_data_generator.py

# Reset a type to new major version
python synthetic_data_generator.py reset direct

# Create persistent minor version
python synthetic_data_generator.py init sequential

# View cache stats
python synthetic_data_generator.py stats direct

# View activity log
python synthetic_data_generator.py log
```

## Output Format

```json
{
  "meta": {
    "type": "sequential",
    "version": "1_0",
    "time": "2026-01-11T20:00:00",
    "elapsed": 45.2,
    "count": 500,
    "config": {
      "batch_size": 25,
      "num_samples": 500,
      "model": "gpt-4o-mini"
    }
  },
  "samples": [
    {
      "id": "sequential-v1_0-001",
      "type": "sequential",
      "version": "1_0",
      "data": {
        "question": "Which gene...",
        "short_answer": "GBA",
        "reasoning": "Step 1: ... Step 2: ...",
        "topic": "Pathophysiology",
        "pattern": "what_treatment_caused_by",
        "entities_used": ["Gaucher disease", "GBA"]
      },
      "status": "ok"
    }
  ],
  "prompt_template": {
    "system": "...",
    "user": "...",
    "entities_used": [...]
  }
}
```

## Sample ID Format

```
{type}-v{major}_{minor}-{sample:03d}

Examples:
- sequential-v1_0-001
- direct-v2_0-042
- parallel-v1_3-100
```

## Workflow: Generate All Types

```bash
# 1. Generate Direct
# Edit config.yml: question_type: direct
python generate_samples.py

# 2. Generate Sequential
# Edit config.yml: question_type: sequential
python generate_samples.py

# 3. Generate Parallel
# Edit config.yml: question_type: parallel
python generate_samples.py
```

## Few-Shot Examples

From BioKGBench/BioHopR development set:

**Direct:**
- "What genetic disorder with hypoparathyroidism, deafness, renal disease causes proteinuria?" → HDR syndrome
- "What is the second cause of dementia after Alzheimer's?" → Vascular dementia

**Sequential:**
- "Which chromosome has the gene encoding the enzyme that inactivates antithrombin?" → Chromosome 11
- "On which chromosome is the gene causing DFNB1?" → Chromosome 13

**Parallel:**
- "What symptom is shared by thyroid hormone resistance and goitre?" → Thyroid enlargement
- "Which keratin gene is associated with both Pachyonychia congenita and Steatocystoma multiplex?" → KRT17

## Dependencies

- `yaml` - Configuration
- `requests` - API calls
- `json` - Data handling

## Files

| File | Purpose |
|------|---------|
| `generate_samples.py` | Main generation script |
| `synthetic_data_generator.py` | Templates, cache, logging |
| `mesh_entity_extractor.py` | MeSH entity extraction |
| `data_generation_prompt.py` | Prompt templates, few-shots |
| `config.yml` | Configuration |
