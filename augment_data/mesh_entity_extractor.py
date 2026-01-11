"""MeSH Entity Extractor - Simple functions for entity extraction with no repeats."""

import json
import random
from pathlib import Path

# MeSH category mapping
MESH_CATEGORIES = {
    "A": "Anatomy", "B": "Organisms", "C": "Diseases", "D": "Chemicals and Drugs",
    "E": "Techniques", "F": "Psychology", "G": "Phenomena", "H": "Disciplines",
    "I": "Social", "J": "Technology", "K": "Humanities", "L": "Information",
    "M": "Named Groups", "N": "Health Care", "V": "Publications", "Z": "Geographicals"
}

# Global state for tracking used entities
_entities_by_cat = {}
_used_ids = set()


def load_mesh_data(path: str = None, seed: int = None):
    """Load and shuffle MeSH data. Call once at start."""
    global _entities_by_cat, _used_ids
    
    path = path or str(Path(__file__).parent / "shallow_query_result_2.json")
    if seed:
        random.seed(seed)
    
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    
    _entities_by_cat = {}
    _used_ids = set()
    
    for item in data:
        terms = item.get("terms", [])
        if not terms:
            continue
        
        entity = {
            "id": item.get("find_from", ""),
            "term": terms[0].title(),
            "definition": item.get("definition", "")[:150]
        }
        
        for cat in item.get("category", []):
            if cat not in _entities_by_cat:
                _entities_by_cat[cat] = []
            _entities_by_cat[cat].append(entity)
    
    # Shuffle all
    for cat in _entities_by_cat:
        random.shuffle(_entities_by_cat[cat])
    
    print(f"Loaded {len(data)} MeSH entities")
    return len(data)


def get_entities(category: str, count: int = 10) -> list:
    """Get unused entities from category. Each entity used only once."""
    global _used_ids
    
    result = []
    for e in _entities_by_cat.get(category, []):
        if e["id"] not in _used_ids:
            result.append(e["term"])
            _used_ids.add(e["id"])
            if len(result) >= count:
                break
    return result


def get_filtered_entities(category: str, keywords: list, count: int = 10) -> list:
    """Get entities matching keywords in term or definition."""
    global _used_ids
    
    result = []
    for e in _entities_by_cat.get(category, []):
        if e["id"] in _used_ids:
            continue
        text = (e["term"] + " " + e["definition"]).lower()
        if any(kw in text for kw in keywords):
            result.append(e["term"])
            _used_ids.add(e["id"])
            if len(result) >= count:
                break
    return result


def get_entity_pool(config: dict = None) -> dict:
    """Get entity pool for generation. Each call returns fresh, unused entities."""
    config = config or {}
    counts = config.get("entity_counts", {})
    default = config.get("default_entity_count", 10)
    
    return {
        "diseases": get_entities("C", counts.get("diseases", default)),
        "chemicals_drugs": get_entities("D", counts.get("chemicals_drugs", default)),
        "anatomy": get_entities("A", counts.get("anatomy", default)),
        "phenomena": get_entities("G", counts.get("phenomena_processes", default)),
        "techniques": get_entities("E", counts.get("techniques_equipment", 5)),
        "organisms": get_entities("B", counts.get("organisms", 5)),
        "genes": get_filtered_entities("D", ["gene", "dna", "rna", "nucleic"], counts.get("genes", default)),
        "proteins": get_filtered_entities("D", ["protein", "enzyme", "ase"], counts.get("proteins", default)),
        "symptoms": get_filtered_entities("C", ["pain", "syndrome", "fever", "failure", "disorder"], counts.get("symptoms", default)),
        "drugs": get_filtered_entities("D", ["drug", "therapeutic", "treatment", "mab", "nib"], counts.get("drugs", default)),
    }


def reset():
    """Reset used entity tracking."""
    global _used_ids
    _used_ids = set()
    for cat in _entities_by_cat:
        random.shuffle(_entities_by_cat[cat])


def get_stats() -> dict:
    """Get usage stats."""
    return {
        "used": len(_used_ids),
        "remaining": {MESH_CATEGORIES.get(c, c): len([e for e in v if e["id"] not in _used_ids]) 
                      for c, v in _entities_by_cat.items() if c in "CDAGEB"}
    }


if __name__ == "__main__":
    load_mesh_data(seed=42)
    
    print("\nBatch 1:")
    pool1 = get_entity_pool({"default_entity_count": 3})
    for k, v in pool1.items():
        print(f"  {k}: {v}")
    
    print("\nBatch 2 (different entities):")
    pool2 = get_entity_pool({"default_entity_count": 3})
    for k, v in pool2.items():
        print(f"  {k}: {v}")
    
    print(f"\nStats: {get_stats()}")
