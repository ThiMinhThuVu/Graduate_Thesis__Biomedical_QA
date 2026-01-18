"""
Generate synthetic biomedical QA samples - ONE type at a time.
===============================================================
Each run generates samples for a single question type only.
"""

import json
import sys
import time
import requests
from pathlib import Path
from datetime import datetime
from math import ceil

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from augment_data.synthetic_data_generator import (
    TEMPLATES, select_template, record_usage, 
    load_cache, init_cache, log, get_stats, list_versions
)
from augment_data import mesh_entity_extractor as mesh

# Direct API call - bypasses serpapi import issue
try:
    from src.api_key import API_LIST_4O, API_LIST_O3
    HAS_API = True
except ImportError as e:
    print(f"Warning: No API keys - {e}")
    HAS_API = False

API_URL = "https://cf.mnapi.com/v1/chat/completions"
_api_idx = 0


def call_llm(messages, model="gpt-4o-mini"):
    """Direct LLM API call - bypasses gpt_request.py serpapi issue."""
    global _api_idx
    
    if not HAS_API:
        return None
    
    API_LIST = API_LIST_4O if "4o-mini" in model else API_LIST_O3
    
    for attempt in range(3):
        try:
            r = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_LIST[_api_idx]}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 4096},
                timeout=120
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            else:
                print(f"API {r.status_code}", end=" ")
        except Exception as e:
            print(f"Err:{e}", end=" ")
        
        _api_idx = (_api_idx + 1) % len(API_LIST)
        time.sleep(2)
    
    return None


def load_config():
    with open(Path(__file__).parent / "config.yml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_json(text):
    if not text:
        return []
    try:
        start, end = text.find('['), text.rfind(']') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except:
        pass
    return []


def get_entities(qtype: str, batch_size: int) -> list:
    """Get starting entities - 1 per question (2 for parallel)."""
    needed = batch_size * 2 if qtype == "parallel" else batch_size
    return mesh.get_entities("C", needed // 2 + 5) + mesh.get_entities("D", needed // 2 + 5)


def build_prompt(qtype: str, num: int, entities: list, template_pattern: str):
    """Build prompt for a SINGLE template pattern."""
    
    all_templates = TEMPLATES.get(qtype, [])
    
    # Find the specific template
    template = None
    for t in all_templates:
        if t["pattern"] == template_pattern:
            template = t
            break
    
    if not template:
        raise ValueError(f"Template '{template_pattern}' not found in '{qtype}'. Available: {[t['pattern'] for t in all_templates]}")
    
    # Build template info string
    template_info = {
        "pattern": template["pattern"],
        "structure": template["structure"],
    }
    if "chain" in template:
        template_info["chain"] = template["chain"]
    if "branches" in template:
        template_info["branches"] = template["branches"]
    if "example" in template:
        template_info["example"] = template["example"]
    
    template_str = json.dumps(template_info, indent=2)
    entities_str = "\n".join(f"- {e}" for e in entities[:num + 10])
    
    rules = {
        "direct_simple": "Use 1 entity as given, find answer directly",
        "direct_hard": "Use multiple entities as given, find answer directly (single-hop)",
        "sequential": "Use 1 entity as START. Answer requires discovering IMPLICIT intermediate steps",
        "parallel": "Use 2 entities to compare"
    }

    system = f"""You are a biomedical QA expert. Generate {num} natural-sounding {qtype} questions following the template pattern.

CRITICAL - WORDING DIVERSITY RULES:
1. DO NOT copy the template structure word-for-word. Use it ONLY as a conceptual guide.
2. Each question MUST use different phrasing, sentence structure, and vocabulary.
3. Vary question openings: mix "Which", "What", "How", "Can you identify", "Name the", "Identify", etc.
4. Rephrase concepts creatively: "causes" → "leads to", "is responsible for", "triggers", "results in"
5. Change sentence flow: active/passive voice, different clause orders, embedded questions
6. The template shows the REASONING CHAIN - not the exact words to use!

Other Rules:
- Questions should sound natural, like real biomedical exam questions written by different professors
- Use the provided entities as starting points
- Avoid repetitive patterns across questions"""

    # Sequential-specific guidance (simplified)
    sequential_hint = ""
    if qtype == "sequential":
        sequential_hint = """

SEQUENTIAL RULES:
- Only the STARTING entity should be explicit in the question
- Intermediate entities must be IMPLICIT (discovered through reasoning)
- Example variations for the SAME reasoning chain:
  • "Which chromosome contains the gene that causes [DISEASE]?"
  • "On what chromosome can we find the gene responsible for [DISEASE]?"
  • "The gene implicated in [DISEASE] is located on which chromosome?"
  • "Identify the chromosomal location of the gene linked to [DISEASE]."
  All follow: DISEASE (given) → GENE (implicit) → CHROMOSOME (answer)
"""

    user = f"""Generate {num} {qtype} biomedical questions using this template:

TEMPLATE: {template_pattern}
{template_str}

ENTITIES TO USE:
{entities_str}

RULE: {rules[qtype]}

⚠️ DIVERSITY REQUIREMENT: Each question MUST have distinctly different wording. Do NOT repeat sentence patterns.
- Use synonyms and alternative phrasings
- Vary grammatical structures (questions, statements-turned-questions, embedded clauses)
- The template is a CONCEPT GUIDE, not a fill-in-the-blank!
{sequential_hint}
OUTPUT FORMAT (JSON array):
[{{"question": "...", "short_answer": "1-3 words", "reasoning": "step-by-step", "topic": "Pathophysiology|Treatment|Diagnosis", "pattern": "{template_pattern}", "entities_used": ["given", "answer"]}}]

Generate {num} diverse questions:"""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_batch(qtype: str, num: int, model: str, cache: dict):
    """Generate one batch."""
    entities = get_entities(qtype, num)
    prompt = build_prompt(qtype, num, entities, cache)
    tokens = (len(prompt[0]["content"]) + len(prompt[1]["content"])) // 4
    
    print(f"    {num}q, {len(entities)} entities, ~{tokens} tokens...", end=" ", flush=True)
    
    if not HAS_API:
        print("(no API)")
        return []
    
    try:
        response = call_llm(prompt, model=model)
        results = parse_json(response)
        
        # Record usage for each result
        for r in results:
            pattern = r.get("pattern")
            topic = r.get("topic")
            template = next((t for t in TEMPLATES[qtype] if t["pattern"] == pattern), TEMPLATES[qtype][0])
            cache = record_usage(qtype, template, topic, cache)
        
        return results
    except Exception as e:
        print(f"Error: {e}")
        log(f"Batch error: {e}", qtype)
        return []


def main():
    config = load_config()
    gen = config.get("generation", {})
    model = config.get("llm", {}).get("model", "gpt-4o-mini")
    
    # Get SINGLE question type
    qtype_config = gen.get("question_type")
    if not qtype_config or qtype_config not in TEMPLATES:
        valid_types = ", ".join(TEMPLATES.keys())
        print(f"ERROR: Invalid 'question_type' in config.yml: '{qtype_config}'")
        print(f"  Valid types: {valid_types}")
        sys.exit(1)
    
    qtype = qtype_config
    num_samples = gen.get("num_samples", 50)
    batch_size = gen.get("batch_size", 25)
    reset_version = gen.get("reset_version", False)
    
    template_pattern = gen.get("template_pattern", None)
    if not template_pattern:
        print("ERROR: template_pattern not set in config.yml")
        print("Run 'python augment_data/synthetic_data_generator.py' to see available patterns")
        return
    
    num_batches = ceil(num_samples / batch_size)
    
    versions = list_versions(qtype)
    print(f"\n{'='*50}")
    print(f"GENERATING: {qtype.upper()} / {template_pattern}")
    print(f"{'='*50}")
    print(f"Existing versions: {versions.get(qtype, [])}")
    
    # Initialize cache
    if reset_version:
        cache = init_cache(qtype, reset=True)
        print(f"Reset to new version: v{cache['version']}")
    else:
        latest = versions.get(qtype, [])
        if latest:
            cache = init_cache(qtype, reset=False)
            print(f"Persistent version: v{cache['version']}")
        else:
            cache = init_cache(qtype, reset=True)
            print(f"First version: v{cache['version']}")
    
    print(f"Samples: {num_samples}, Batch: {batch_size}, Batches: {num_batches}")
    print(f"API available: {HAS_API}")
    
    # Load MeSH entities
    if config.get("entity_source") == "mesh":
        mesh.load_mesh_data(seed=int(datetime.now().timestamp()) % 10000)
        
        # Exclude entities already used in previous minor versions
        prev_used = cache.get("used_entity_ids", [])
        if prev_used:
            mesh.exclude_from_cache(prev_used)
    
    log(f"Started generation: {num_samples} samples, batch={batch_size}", qtype)
    
    # Generate
    samples = []
    sample_prompt = None  # Save one example prompt
    start = datetime.now()
    
    for b in range(num_batches):
        remaining = num_samples - len(samples)
        batch = min(batch_size, remaining)
        if batch <= 0:
            break
        
        print(f"  Batch {b+1}/{num_batches}:", end="")
        
        # Get entities and build prompt
        entities = get_entities(qtype, batch)
        prompt = build_prompt(qtype, batch, entities, template_pattern)
        
        # Save first prompt as sample
        if sample_prompt is None:
            sample_prompt = {
                "system": prompt[0]["content"],
                "user": prompt[1]["content"],
                "entities_used": entities[:batch + 5]
            }
        
        tokens = (len(prompt[0]["content"]) + len(prompt[1]["content"])) // 4
        print(f"    {batch}q, {len(entities)} entities, ~{tokens} tokens...", end=" ", flush=True)
        
        if not HAS_API:
            print("(no API)")
            continue
        
        try:
            response = call_llm(prompt, model=model)
            results = parse_json(response)
            
            for r in results:
                pattern = r.get("pattern")
                topic = r.get("topic")
                template = next((t for t in TEMPLATES[qtype] if t["pattern"] == pattern), TEMPLATES[qtype][0])
                cache = record_usage(qtype, template, topic, cache)
            
            if results:
                print(f"✓ {len(results)}")
                for r in results:
                    # ID format: type-vX_Y-NNN (e.g., sequential-v1_0-001)
                    sample_num = len(samples) + 1
                    samples.append({
                        "id": f"{qtype}-v{cache['version']}-{sample_num:03d}",
                        "type": qtype,
                        "version": cache["version"],
                        "data": r,
                        "status": "ok"
                    })
                
                # Stop if we've reached target
                if len(samples) >= num_samples:
                    print(f"  → Reached target ({len(samples)} ≥ {num_samples}), stopping.")
                    break
            else:
                print("✗")
        except Exception as e:
            print(f"Error: {e}")
            log(f"Batch error: {e}", qtype)
    
    elapsed = (datetime.now() - start).total_seconds()
    log(f"Completed: {len(samples)} samples in {elapsed:.1f}s", qtype)
    
    # Update cache with used entity IDs for future runs
    cache["used_entity_ids"] = mesh.get_used_ids()
    from augment_data.synthetic_data_generator import save_cache
    save_cache(qtype, cache)
    
    # Save output - in generated/ subdirectory
    version = cache["version"]
    output_file = f"{qtype}_{template_pattern}_v{version}.json"
    output = Path(__file__).parent / "output" / "generated" / output_file
    output.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "meta": {
            "type": qtype,
            "template_pattern": template_pattern,
            "version": version,
            "time": datetime.now().isoformat(),
            "elapsed": elapsed,
            "count": len(samples),
            "config": {
                "batch_size": batch_size,
                "num_samples": num_samples,
                "model": model
            }
        },
        "samples": samples,
        "prompt_template": sample_prompt  # Save the prompt used
    }
    
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"DONE: {qtype} v{version}")
    print(f"{'='*50}")
    print(f"Samples: {len(samples)}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Output: {output}")
    print(f"Prompt saved: yes")
    
    stats = get_stats(qtype)
    print(f"\nCache v{version}:")
    print(f"  Total: {stats['total']}")
    print(f"  Patterns: {stats['patterns']}")
    print(f"  Topics: {stats['topics']}")


if __name__ == "__main__":
    main()
