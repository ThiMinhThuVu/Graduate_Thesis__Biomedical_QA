"""
Synthetic Biomedical Question Generation - Templates and Versioned Cache
=========================================================================
- Versioned cache files per question type
- Activity logging
- Single type generation only
"""

import json
import random
from pathlib import Path
from datetime import datetime

# Directories
OUTPUT_DIR = Path(__file__).parent / "output"
CACHE_DIR = OUTPUT_DIR / "cache"
LOG_FILE = OUTPUT_DIR / "generation.log"

# =============================================================================
# TEMPLATES
# =============================================================================

TEMPLATES = {
    "direct": [
        # === ORIGINAL TEMPLATES ===
        {"pattern": "which_specialty", "structure": "Which medical specialty is most likely to treat patients with [DISEASE]?", "wh_word": "Which", "topic": "Treatment", "example": "Which medical specialty is most likely to treat patients with Gaucher disease?"},
        {"pattern": "what_risk_factor", "structure": "What is a common risk factor for both [DISEASE_A] and [DISEASE_B]?", "wh_word": "What", "topic": "Prevention", "example": "What is a common risk factor for both diabetes and heart disease?"},
        {"pattern": "what_type", "structure": "What type of [ENTITY] is associated with [CONDITION]?", "wh_word": "What", "topic": "Diagnosis", "example": "What type of enzyme is associated with phenylketonuria?"},
        {"pattern": "mechanism", "structure": "How does [CHEMICAL/PROTEIN] contribute to the development of [DISEASE]?", "wh_word": "How", "topic": "Pathophysiology", "example": "How does alpha-synuclein contribute to the development of Parkinson's disease?"},
        {"pattern": "how_regulate", "structure": "How do [ENTITY_PLURAL] contribute to the regulation of [PROCESS]?", "wh_word": "How", "topic": "Pathophysiology", "example": "How do cytokines contribute to the regulation of inflammation?"},
        {"pattern": "on_chromosome", "structure": "On which chromosome is the [GENE] gene, known to cause [DISEASE], located?", "wh_word": "On which", "topic": "Pathophysiology", "example": "On which chromosome is the CFTR gene, known to cause cystic fibrosis, located?"},
        {"pattern": "is_notifiable", "structure": "Is [DISEASE] a notifiable disease?", "wh_word": "Is", "topic": "Prevention", "example": "Is tuberculosis a notifiable disease?"},
        {"pattern": "can_lead_to", "structure": "Can [CONDITION] lead to [SYMPTOM/COMPLICATION]?", "wh_word": "Can", "topic": "Diagnosis", "example": "Can diabetes lead to neuropathy?"},
        {"pattern": "does_include", "structure": "Does [SYNDROME] include [SYMPTOM] as a symptom?", "wh_word": "Does", "topic": "Diagnosis", "example": "Does Marfan syndrome include aortic dilation as a symptom?"},
        
        # === NEW: From misclassified patterns ===
        {"pattern": "in_what_year", "structure": "In what year was [TREATMENT/DISCOVERY] for [DISEASE] first [APPROVED/DESCRIBED]?", "wh_word": "In what", "topic": "Treatment", "example": "In what year was insulin for diabetes first approved?"},
        {"pattern": "eponymous_disorder", "structure": "Which [CONDITION] is named after the [PROFESSION] who first described it?", "wh_word": "Which", "topic": "Diagnosis", "example": "Which disease is named after the neurologist who first described it?"},
        {"pattern": "anatomy_affected", "structure": "Which part of the [ORGAN] is primarily affected by [CONDITION]?", "wh_word": "Which", "topic": "Pathophysiology", "example": "Which part of the eye is primarily affected by glaucoma?"},
        {"pattern": "how_many_count", "structure": "How many [UNITS] are in the [PROTEIN/ENZYME] associated with [DISEASE]?", "wh_word": "How many", "topic": "Pathophysiology", "example": "How many amino acid residues are in the enzyme whose mutations cause PKU?"},
        {"pattern": "which_organization", "structure": "Which organization [PUBLISHES/APPROVES/REGULATES] [TREATMENT/JOURNAL] for [DISEASE]?", "wh_word": "Which", "topic": "Treatment", "example": "Which organization approves treatments for rare diseases?"},
        {"pattern": "medication_for_symptom", "structure": "Which medication is used to treat the [SYMPTOM] associated with [SYNDROME]?", "wh_word": "Which", "topic": "Treatment", "example": "Which medication is used to treat the nerve pain associated with shingles?"},
        {"pattern": "cytokine_role", "structure": "Which cytokine is elevated in [CONDITION] and acts as [ROLE]?", "wh_word": "Which", "topic": "Pathophysiology", "example": "Which cytokine is elevated in inflammation and acts as a pro-inflammatory mediator?"},
    ],
    "sequential": [
        # === ORIGINAL TEMPLATES ===
        {"pattern": "what_treatment_caused_by", "structure": "What [DRUG?] is used to treat the [DISEASE?] caused by [GENE] mutations?", "chain": "[GENE] → [DISEASE] → [DRUG]", "wh_word": "What", "topic": "Treatment", "example": "What drug is used to treat the disease caused by GBA mutations?"},
        {"pattern": "what_pathway_disrupted", "structure": "What [PATHWAY?] is disrupted in the [DISEASE?] characterized by [SYMPTOM]?", "chain": "[SYMPTOM] → [DISEASE] → [PATHWAY]", "wh_word": "What", "topic": "Pathophysiology", "example": "What pathway is disrupted in the disease characterized by hepatomegaly?"},
        {"pattern": "on_chromosome_causes", "structure": "On which chromosome is the [GENE?] that causes [DISEASE?] characterized by [SYMPTOM]?", "chain": "[SYMPTOM] → [DISEASE] → [GENE] → [CHROMOSOME]", "wh_word": "On which", "topic": "Pathophysiology", "example": "On which chromosome is the gene that causes the disease with tremor?"},
        {"pattern": "is_same_gene", "structure": "Is the deficiency in [PROTEIN?] caused by the same gene that is mutated in [DISEASE]?", "chain": "[DISEASE] → [GENE] → [PROTEIN deficiency]", "wh_word": "Is", "topic": "Diagnosis", "example": "Is glucocerebrosidase deficiency caused by the same gene mutated in Gaucher disease?"},
        {"pattern": "can_mutation_cause", "structure": "Can a mutation in the [GENE?] that encodes [PROTEIN] cause [DISEASE]?", "chain": "[PROTEIN] → [GENE] → [DISEASE]", "wh_word": "Can", "topic": "Pathophysiology", "example": "Can a mutation in the gene encoding dystrophin cause muscular dystrophy?"},
        {"pattern": "do_mutations_lead", "structure": "Do mutations in [GENE?] lead to [DISEASE?] that also causes [SYMPTOM]?", "chain": "[GENE] → [DISEASE] → [SYMPTOM]", "wh_word": "Do", "topic": "Pathophysiology", "example": "Do mutations in HTT lead to a disease that also causes chorea?"},
        
        # === NEW: From misclassified patterns ===
        {"pattern": "cell_organelle_affected", "structure": "What is the [ORGANELLE?] of [CELL_TYPE] that is affected in [DISEASE]?", "chain": "[DISEASE] → [CELL_TYPE] → [ORGANELLE]", "wh_word": "What", "topic": "Pathophysiology", "example": "What is the organelle of muscle cells that is affected in mitochondrial myopathy?"},
        {"pattern": "vaccine_for_disease", "structure": "What is the [VACCINE_TYPE?] vaccine that has shown results for [DISEASE] caused by [PATHOGEN]?", "chain": "[PATHOGEN] → [DISEASE] → [VACCINE]", "wh_word": "What", "topic": "Treatment", "example": "What is the live-attenuated vaccine for the disease caused by Rift Valley fever virus?"},
        {"pattern": "treatment_for_condition", "structure": "What is the only effective treatment for [CONDITION] that occurs in [POPULATION]?", "chain": "[POPULATION] → [CONDITION] → [TREATMENT]", "wh_word": "What", "topic": "Treatment", "example": "What is the only effective treatment for neural tube defects?"},
        {"pattern": "complication_anatomy", "structure": "What is a potential complication of [DISEASE] that involves [ANATOMY_PROCESS]?", "chain": "[DISEASE] → [COMPLICATION] → [ANATOMY]", "wh_word": "What", "topic": "Diagnosis", "example": "What is a potential complication of Crohn's disease that involves the intestinal wall?"},
        {"pattern": "protein_from_gene", "structure": "What protein encoded by [GENE] is essential for [PROCESS] in [ORGANELLE]?", "chain": "[GENE] → [PROTEIN] → [PROCESS]", "wh_word": "What", "topic": "Pathophysiology", "example": "What protein encoded by POLG is essential for mitochondrial DNA replication?"},
        {"pattern": "gene_causes_symptoms", "structure": "Which gene mutation causes a disease that leads to [SYMPTOM_A] and [SYMPTOM_B]?", "chain": "[SYMPTOMS] → [DISEASE] → [GENE]", "wh_word": "Which", "topic": "Pathophysiology", "example": "Which gene mutation causes a disease that leads to hearing loss and dystonia?"},
        {"pattern": "enzyme_loss_symptom", "structure": "Which enzyme's loss of function, due to mutations in [GENE], leads to [SYMPTOM]?", "chain": "[GENE] → [ENZYME] → [SYMPTOM]", "wh_word": "Which", "topic": "Pathophysiology", "example": "Which enzyme's loss of function, due to PANK2 mutations, leads to spasticity?"},
        {"pattern": "treatment_dual_role", "structure": "What is a treatment for [SYMPTOM] that also plays a role in [SYSTEM/PROCESS]?", "chain": "[SYMPTOM] → [TREATMENT] → [ROLE]", "wh_word": "What", "topic": "Treatment", "example": "What is a treatment for taste disorders that also supports the immune system?"},
        {"pattern": "gene_chromosome_encodes", "structure": "Which gene on [CHROMOSOME] encodes [PROTEIN_TYPE] for [PROCESS]?", "chain": "[CHROMOSOME] → [GENE] → [PROTEIN] → [PROCESS]", "wh_word": "Which", "topic": "Pathophysiology", "example": "Which gene on the X chromosome encodes a guanine nucleotide exchange factor?"},
        {"pattern": "condition_comorbid", "structure": "What psychological condition can develop in patients with [CONDITION] and is prevalent in [DISORDER]?", "chain": "[DISORDER] → [CONDITION] → [COMORBIDITY]", "wh_word": "What", "topic": "Diagnosis", "example": "What condition develops in patients with body image disturbance and is common in anorexia?"},
        {"pattern": "triplet_repeat_gene", "structure": "Which syndrome is caused by expansion of [REPEAT_TYPE] in [GENE] on [CHROMOSOME]?", "chain": "[CHROMOSOME] → [GENE] → [REPEAT] → [SYNDROME]", "wh_word": "Which", "topic": "Pathophysiology", "example": "Which syndrome is caused by CGG triplet repeat expansion in FMR1 on the X chromosome?"},
        {"pattern": "disease_also_known_as", "structure": "What is the common name of [DISEASE] that is also caused by mutations in [GENE]?", "chain": "[GENE] → [DISEASE] → [COMMON_NAME]", "wh_word": "What", "topic": "Diagnosis", "example": "What is the common name of the disease caused by GBA mutations, also known as familial Parkinsonism?"},
        {"pattern": "disease_class_count", "structure": "The disease that [CONDITION_A] resembles is classified as a [CLASS]. How many types exist?", "chain": "[CONDITION] → [SIMILAR_DISEASE] → [CLASS] → [COUNT]", "wh_word": "How many", "topic": "Diagnosis", "example": "The disease that pseudo-Hurler resembles is a mucopolysaccharidosis. How many types exist?"},
        {"pattern": "enzyme_absent_compound", "structure": "What enzyme's activity is absent in [DISEASE] that leads to high levels of [COMPOUND]?", "chain": "[DISEASE] → [ENZYME_ABSENT] → [COMPOUND_HIGH]", "wh_word": "What", "topic": "Pathophysiology", "example": "What enzyme's activity is absent in molybdenum cofactor deficiency leading to high uric acid?"},
        {"pattern": "gene_affects_function", "structure": "Which chromosome contains genes that affect [CELL_TYPE] function when mutated in [DISEASE]?", "chain": "[DISEASE] → [GENES] → [CHROMOSOME] → [FUNCTION]", "wh_word": "Which", "topic": "Pathophysiology", "example": "Which chromosome contains genes that affect lymphocyte function in immunodeficiency?"},
        {"pattern": "genes_collagen_type", "structure": "The genes affected in [DISEASE] are associated with variants of which type of [PROTEIN]?", "chain": "[DISEASE] → [GENES] → [PROTEIN_TYPE]", "wh_word": "Which", "topic": "Pathophysiology", "example": "The genes affected in LGMD are associated with variants of which type of collagen?"},
    ],
    "parallel": [
        # === ORIGINAL TEMPLATES ===
        {"pattern": "or_choice_gene", "structure": "Is [DISEASE] caused by mutations in [GENE_A] or [GENE_B]?", "branches": ["[DISEASE] → [GENE_A]?", "[DISEASE] → [GENE_B]?"], "conjunction": "or", "wh_word": "Is", "topic": "Pathophysiology", "example": "Is muscular dystrophy caused by mutations in DMD or DMPK?"},
        {"pattern": "or_choice_treatment", "structure": "Which is more effective for treating [DISEASE], [DRUG_A] or [DRUG_B]?", "branches": ["[DISEASE] ← [DRUG_A]", "[DISEASE] ← [DRUG_B]"], "conjunction": "or", "wh_word": "Which", "topic": "Treatment", "example": "Which is more effective for treating Parkinson's, levodopa or pramipexole?"},
        {"pattern": "difference", "structure": "What [SYMPTOM?] occurs in [DISEASE_A] but not in [DISEASE_B]?", "branches": ["[DISEASE_A] → symptoms", "[DISEASE_B] → symptoms"], "conjunction": "but not", "wh_word": "What", "topic": "Diagnosis", "example": "What symptom occurs in Gaucher disease but not in Fabry disease?"},
        {"pattern": "dual_chain_family", "structure": "Is the [GENE?] mutated in [DISEASE_A] from the same family as the [GENE?] in [DISEASE_B]?", "branches": ["[DISEASE_A] → gene → family", "[DISEASE_B] → gene → family"], "conjunction": "same family", "wh_word": "Is", "topic": "Pathophysiology", "example": "Is the gene mutated in Tay-Sachs from the same family as the gene in Sandhoff disease?"},
        
        # === NEW: Comparison patterns ===
        {"pattern": "shared_symptom", "structure": "What is a common symptom shared by both [DISEASE_A] and [DISEASE_B]?", "branches": ["[DISEASE_A] → symptoms", "[DISEASE_B] → symptoms"], "conjunction": "shared", "wh_word": "What", "topic": "Diagnosis", "example": "What is a common symptom shared by both Parkinson's and essential tremor?"},
        {"pattern": "same_mechanism", "structure": "Do [DRUG_A] and [DRUG_B] work through the same mechanism of action?", "branches": ["[DRUG_A] → mechanism", "[DRUG_B] → mechanism"], "conjunction": "same", "wh_word": "Do", "topic": "Treatment", "example": "Do aspirin and ibuprofen work through the same mechanism of action?"},
        {"pattern": "inheritance_compare", "structure": "Are both [DISEASE_A] and [DISEASE_B] inherited in the same pattern?", "branches": ["[DISEASE_A] → inheritance", "[DISEASE_B] → inheritance"], "conjunction": "same", "wh_word": "Are", "topic": "Pathophysiology", "example": "Are both Huntington's disease and cystic fibrosis inherited in the same pattern?"},
        {"pattern": "pathway_overlap", "structure": "Do [DISEASE_A] and [DISEASE_B] involve the same metabolic pathway?", "branches": ["[DISEASE_A] → pathway", "[DISEASE_B] → pathway"], "conjunction": "same", "wh_word": "Do", "topic": "Pathophysiology", "example": "Do phenylketonuria and tyrosinemia involve the same metabolic pathway?"},
        {"pattern": "enzyme_compare", "structure": "Is the enzyme deficient in [DISEASE_A] from the same class as in [DISEASE_B]?", "branches": ["[DISEASE_A] → enzyme → class", "[DISEASE_B] → enzyme → class"], "conjunction": "same class", "wh_word": "Is", "topic": "Pathophysiology", "example": "Is the enzyme deficient in Gaucher from the same class as in Fabry disease?"},
        {"pattern": "prognosis_compare", "structure": "Which has a better prognosis, [DISEASE_A] or [DISEASE_B]?", "branches": ["[DISEASE_A] → prognosis", "[DISEASE_B] → prognosis"], "conjunction": "or", "wh_word": "Which", "topic": "Diagnosis", "example": "Which has a better prognosis, Hodgkin's lymphoma or pancreatic cancer?"},
        {"pattern": "risk_factor_shared", "structure": "What risk factor is common to both [DISEASE_A] and [DISEASE_B]?", "branches": ["[DISEASE_A] → risk factors", "[DISEASE_B] → risk factors"], "conjunction": "common", "wh_word": "What", "topic": "Prevention", "example": "What risk factor is common to both lung cancer and COPD?"},
        {"pattern": "anatomy_compare", "structure": "Do [DISEASE_A] and [DISEASE_B] primarily affect the same organ system?", "branches": ["[DISEASE_A] → organ", "[DISEASE_B] → organ"], "conjunction": "same", "wh_word": "Do", "topic": "Diagnosis", "example": "Do multiple sclerosis and ALS primarily affect the same organ system?"},
        {"pattern": "biomarker_shared", "structure": "Is [BIOMARKER] elevated in both [DISEASE_A] and [DISEASE_B]?", "branches": ["[DISEASE_A] → biomarker level", "[DISEASE_B] → biomarker level"], "conjunction": "both", "wh_word": "Is", "topic": "Diagnosis", "example": "Is CRP elevated in both rheumatoid arthritis and systemic lupus erythematosus?"},
        {"pattern": "side_effect_compare", "structure": "Which drug has fewer side effects for treating [CONDITION], [DRUG_A] or [DRUG_B]?", "branches": ["[DRUG_A] → side effects", "[DRUG_B] → side effects"], "conjunction": "or", "wh_word": "Which", "topic": "Treatment", "example": "Which drug has fewer side effects for treating hypertension, ACE inhibitors or beta blockers?"},
        {"pattern": "age_onset_compare", "structure": "Which typically presents earlier in life, [DISEASE_A] or [DISEASE_B]?", "branches": ["[DISEASE_A] → age of onset", "[DISEASE_B] → age of onset"], "conjunction": "or", "wh_word": "Which", "topic": "Diagnosis", "example": "Which typically presents earlier in life, type 1 diabetes or type 2 diabetes?"},
        {"pattern": "transmission_compare", "structure": "Are [DISEASE_A] and [DISEASE_B] transmitted through the same route?", "branches": ["[DISEASE_A] → transmission", "[DISEASE_B] → transmission"], "conjunction": "same", "wh_word": "Are", "topic": "Prevention", "example": "Are hepatitis B and hepatitis C transmitted through the same route?"},
    ]
}


# =============================================================================
# LOGGING
# =============================================================================

def log(message: str, qtype: str = None):
    """Append message to log file."""
    LOG_FILE.parent.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{qtype}]" if qtype else ""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {prefix} {message}\n")


# =============================================================================
# VERSIONED CACHE - Each type tracked independently
# =============================================================================

def get_cache_path(qtype: str, version: str) -> Path:
    """Get cache file path for type and version."""
    return CACHE_DIR / f"{qtype}_v{version}.json"


def find_latest_version(qtype: str) -> str:
    """Find the latest version number for a type (e.g., '1', '2', '2_1', '2_2')."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    versions = []
    for f in CACHE_DIR.glob(f"{qtype}_v*.json"):
        # Extract version from filename like "direct_v2_1.json"
        v = f.stem.replace(f"{qtype}_v", "")
        versions.append(v)
    
    if not versions:
        return None
    
    # Sort versions (handle both "2" and "2_1" formats)
    def version_key(v):
        parts = v.split("_")
        return [int(p) for p in parts]
    
    versions.sort(key=version_key)
    return versions[-1]


def get_next_version(qtype: str, reset: bool = False) -> str:
    """
    Get next version number.
    - reset=True: Increment major version (1 → 2)
    - reset=False: Increment minor version (1_0 → 1_1 → 1_2)
    """
    latest = find_latest_version(qtype)
    
    if latest is None:
        return "1_0"  # First version: 1_0
    
    if reset:
        # New major version: 1_x → 2_0, 2_x → 3_0
        major = int(latest.split("_")[0])
        return f"{major + 1}_0"
    else:
        # Persistent minor version: 1_0 → 1_1 → 1_2
        major, minor = latest.split("_") if "_" in latest else (latest, "-1")
        return f"{major}_{int(minor) + 1}"


def load_cache(qtype: str, version: str = None) -> dict:
    """Load cache for type. If version not specified, use latest."""
    if version is None:
        version = find_latest_version(qtype)
    
    if version is None:
        return {"version": "1_0", "patterns": {}, "topics": {}, "total": 0, "used_entity_ids": [], "created": datetime.now().isoformat()}
    
    cache_path = get_cache_path(qtype, version)
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
            # Ensure used_entity_ids exists for backward compatibility
            if "used_entity_ids" not in cache:
                cache["used_entity_ids"] = []
            return cache
    
    return {"version": version, "patterns": {}, "topics": {}, "total": 0, "used_entity_ids": [], "created": datetime.now().isoformat()}


def save_cache(qtype: str, cache: dict):
    """Save cache for type."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    version = cache.get("version", "1")
    cache_path = get_cache_path(qtype, version)
    cache["updated"] = datetime.now().isoformat()
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def init_cache(qtype: str, reset: bool = False) -> dict:
    """
    Initialize cache for type. Creates new version if needed.
    For persistent versions (reset=False): inherits used_entity_ids from previous version.
    For reset versions (reset=True): starts fresh with empty used_entity_ids.
    """
    # Load previous version's used entities for persistent versioning
    prev_used_entities = []
    if not reset:
        prev_version = find_latest_version(qtype)
        if prev_version:
            prev_cache = load_cache(qtype, prev_version)
            prev_used_entities = prev_cache.get("used_entity_ids", [])
    
    version = get_next_version(qtype, reset=reset)
    cache = {
        "version": version,
        "type": qtype,
        "patterns": {},
        "topics": {},
        "total": 0,
        "used_entity_ids": prev_used_entities,  # Inherit from previous minor version
        "created": datetime.now().isoformat()
    }
    save_cache(qtype, cache)
    
    action = "reset" if reset else "persistent"
    inherited = f" (inherited {len(prev_used_entities)} used entities)" if prev_used_entities else ""
    log(f"Cache initialized v{version} ({action}){inherited}", qtype)
    
    return cache


# =============================================================================
# TEMPLATE FUNCTIONS
# =============================================================================

def select_template(qtype: str, cache: dict = None) -> dict:
    """Select a template, preferring less-used patterns."""
    templates = TEMPLATES.get(qtype, [])
    if not templates:
        raise ValueError(f"No templates for: {qtype}")
    
    if cache is None:
        cache = load_cache(qtype)
    
    scored = []
    for t in templates:
        usage = cache["patterns"].get(t["pattern"], 0)
        score = -usage + random.random()
        scored.append((score, t))
    
    scored.sort(reverse=True)
    return scored[0][1]


def record_usage(qtype: str, template: dict, topic: str = None, cache: dict = None):
    """Record template usage to cache."""
    if cache is None:
        cache = load_cache(qtype)
    
    pattern = template["pattern"]
    cache["patterns"][pattern] = cache["patterns"].get(pattern, 0) + 1
    
    topic = topic or template.get("topic", "Unknown")
    cache["topics"][topic] = cache["topics"].get(topic, 0) + 1
    cache["total"] += 1
    
    save_cache(qtype, cache)
    return cache


def get_stats(qtype: str = None) -> dict:
    """Get stats for type or all types."""
    if qtype:
        version = find_latest_version(qtype)
        if version:
            return load_cache(qtype, version)
        return None
    
    # All types
    stats = {}
    for t in TEMPLATES.keys():
        version = find_latest_version(t)
        if version:
            stats[t] = load_cache(t, version)
    return stats


def list_versions(qtype: str = None) -> dict:
    """List all versions for type(s)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    types = [qtype] if qtype else TEMPLATES.keys()
    result = {}
    
    for t in types:
        versions = []
        for f in sorted(CACHE_DIR.glob(f"{t}_v*.json")):
            v = f.stem.replace(f"{t}_v", "")
            versions.append(v)
        result[t] = versions
    
    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    args = sys.argv[1:]
    
    if not args:
        # Show status
        print("TEMPLATES:")
        for qtype, templates in TEMPLATES.items():
            print(f"  {qtype}: {len(templates)} templates")
        
        print("\nVERSIONS:")
        for qtype, versions in list_versions().items():
            latest = versions[-1] if versions else "none"
            print(f"  {qtype}: {versions} (latest: v{latest})")
        
        print(f"\nLog: {LOG_FILE}")
        
    elif args[0] == "reset" and len(args) > 1:
        qtype = args[1]
        if qtype in TEMPLATES:
            cache = init_cache(qtype, reset=True)
            print(f"Created new version: {qtype} v{cache['version']}")
        else:
            print(f"Unknown type: {qtype}. Use: direct, sequential, parallel")
            
    elif args[0] == "init" and len(args) > 1:
        qtype = args[1]
        if qtype in TEMPLATES:
            reset = "--reset" in args
            cache = init_cache(qtype, reset=reset)
            print(f"Initialized: {qtype} v{cache['version']}")
        else:
            print(f"Unknown type: {qtype}")
            
    elif args[0] == "stats":
        qtype = args[1] if len(args) > 1 else None
        stats = get_stats(qtype)
        print(json.dumps(stats, indent=2))
        
    elif args[0] == "log":
        if LOG_FILE.exists():
            print(LOG_FILE.read_text()[-2000:])  # Last 2000 chars
        else:
            print("No log file yet")
            
    else:
        print("Usage:")
        print("  python synthetic_data_generator.py          # Show status")
        print("  python synthetic_data_generator.py reset <type>   # New major version")
        print("  python synthetic_data_generator.py init <type>    # Persistent version")
        print("  python synthetic_data_generator.py stats [type]   # Show stats")
        print("  python synthetic_data_generator.py log            # Show log")
