"""Generate synthetic biomedical QA samples using MeSH entities (large batch support)."""

import json
import sys
from pathlib import Path
from datetime import datetime
from math import ceil

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from mesh_template.data_generation_prompt import prompt_template_question, FEW_SHOT_EXAMPLES, BIOMEDICAL_CONTEXT
from mesh_template.synthetic_data_generator import TEMPLATES
from mesh_template import mesh_entity_extractor as mesh

try:
    from src.gpt_request import request_api
    HAS_API = True
except ImportError as e:
    print(f"Warning: Cannot import gpt_request: {e}")
    HAS_API = False


def load_config():
    with open(Path(__file__).parent / "config.yml") as f:
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


def build_prompt(qtype, num, entities):
    examples = json.dumps(FEW_SHOT_EXAMPLES.get(qtype, [])[:2], indent=2)
    templates = json.dumps(TEMPLATES.get(qtype, [])[:5], indent=2)
    entities_str = json.dumps(entities, indent=2)
    
    system = f"""You are a biomedical QA generator. Generate {num} unique {qtype} questions.
RULES: 
- Each entity used ONLY ONCE across all questions
- Factually accurate biomedical information only
- Short answer: 1-3 words maximum
- Cover diverse topics: Pathophysiology, Treatment, Diagnosis, Prevention"""

    user = f"""Generate exactly {num} unique {qtype} biomedical questions.

## Templates (for structure inspiration):
{templates}

## ENTITY POOL (use each entity only ONCE):
{entities_str}

## Examples:
{examples}

{BIOMEDICAL_CONTEXT}

## Output Format (JSON array of {num} objects):
[
  {{"question": "...", "short_answer": "1-3 words", "reasoning": "...", "topic": "Pathophysiology|Treatment|Diagnosis|Prevention", "entities_used": ["e1", "e2"]}},
  ...
]

Generate {num} questions now:"""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_batch(qtype, num, config, entities):
    """Generate a single batch of questions."""
    model = config.get("llm", {}).get("model", "gpt-4o-mini")
    
    if not HAS_API:
        return []
    
    try:
        response = request_api(build_prompt(qtype, num, entities), model=model)
        return parse_json(response)
    except Exception as e:
        print(f"Error: {e}")
        return []


def main():
    config = load_config()
    gen = config.get("generation", {})
    
    num_per_type = gen.get("num_per_type", 500)
    batch_size = gen.get("batch_size", 50)
    qtypes = ["direct", "sequential", "parallel"] if gen.get("question_types") == "all" else gen.get("question_types", ["sequential"])
    
    # Calculate batches needed
    num_batches = ceil(num_per_type / batch_size)
    total_api_calls = num_batches * len(qtypes)
    
    # Load MeSH entities
    if config.get("entity_source") == "mesh":
        seed = int(datetime.now().timestamp()) % 10000
        mesh.load_mesh_data(seed=seed)
    
    print(f"\n{'='*60}")
    print(f"LARGE BATCH GENERATION")
    print(f"{'='*60}")
    print(f"Samples per type: {num_per_type}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per type: {num_batches}")
    print(f"Question types: {qtypes}")
    print(f"Total API calls: {total_api_calls}")
    print(f"API available: {HAS_API}")
    
    all_samples = []
    entity_pools_used = []
    start_time = datetime.now()
    
    for qtype in qtypes:
        print(f"\n{'='*40}")
        print(f"{qtype.upper()} - {num_per_type} samples")
        print(f"{'='*40}")
        
        type_samples = []
        
        for batch_num in range(num_batches):
            # Calculate batch size (last batch may be smaller)
            remaining = num_per_type - len(type_samples)
            current_batch = min(batch_size, remaining)
            
            if current_batch <= 0:
                break
            
            # Get fresh entity pool for this batch
            if config.get("entity_source") == "mesh":
                entities = mesh.get_entity_pool(config.get("mesh_entities", {}))
                entity_count = sum(len(v) for v in entities.values())
            else:
                entities = config.get("entities", {})
                entity_count = sum(len(v) for v in entities.values())
            
            print(f"  Batch {batch_num+1}/{num_batches}: {current_batch} questions, {entity_count} entities...", end=" ", flush=True)
            
            if not HAS_API:
                print("(no API)")
                continue
            
            results = generate_batch(qtype, current_batch, config, entities)
            
            if results:
                print(f"✓ got {len(results)}")
                for i, r in enumerate(results):
                    type_samples.append({
                        "id": f"{qtype}_{len(type_samples)+1}",
                        "question_type": qtype,
                        "batch": batch_num + 1,
                        "generated": r,
                        "status": "success"
                    })
                entity_pools_used.append({"type": qtype, "batch": batch_num+1, "entities": entities})
            else:
                print("✗ failed")
        
        print(f"  → {qtype}: {len(type_samples)} samples generated")
        all_samples.extend(type_samples)
    
    # Stats
    elapsed = (datetime.now() - start_time).total_seconds()
    success = [s for s in all_samples if s["status"] == "success"]
    
    entity_usage = {}
    topic_dist = {}
    for s in success:
        for e in s.get("generated", {}).get("entities_used", []):
            entity_usage[e] = entity_usage.get(e, 0) + 1
        topic = s.get("generated", {}).get("topic", "Unknown")
        topic_dist[topic] = topic_dist.get(topic, 0) + 1
    
    duplicates = {k: v for k, v in entity_usage.items() if v > 1}
    
    results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "num_per_type": num_per_type,
            "batch_size": batch_size,
            "types": qtypes,
            "elapsed_seconds": elapsed
        },
        "samples": all_samples,
        "stats": {
            "total": len(all_samples),
            "success": len(success),
            "by_type": {qtype: sum(1 for s in all_samples if s["question_type"] == qtype and s["status"] == "success") for qtype in qtypes},
            "unique_entities": len(entity_usage),
            "topic_distribution": topic_dist,
            "duplicate_count": len(duplicates)
        }
    }
    
    # Include templates if configured
    if gen.get("include_templates", True):
        results["templates"] = TEMPLATES
    
    # Save
    output = Path(__file__).parent / gen.get("output", "output/generated_samples.json")
    output.parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Successful: {len(success)}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Unique entities: {len(entity_usage)}")
    print(f"Duplicates: {len(duplicates)}")
    
    print(f"\nBy type:")
    for qtype in qtypes:
        count = results["stats"]["by_type"][qtype]
        print(f"  - {qtype}: {count}")
    
    print(f"\nTopic distribution:")
    for topic, count in sorted(topic_dist.items(), key=lambda x: -x[1]):
        print(f"  - {topic}: {count}")
    
    if success:
        print(f"\nSample questions:")
        for s in success[:3]:
            q = s["generated"]
            print(f"  [{s['question_type']}] {q.get('question', '')[:70]}...")
    
    print(f"\n✓ Saved to: {output}")
    print(f"  File size: {output.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
