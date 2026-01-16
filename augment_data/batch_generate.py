#!/usr/bin/env python3
"""
Batch generation script - generates equal samples per template with RESUME support.
Usage: python batch_generate.py
"""

import yaml
import subprocess
import json
from pathlib import Path
import time

# Configuration
GENERATIONS = {
    "direct_simple": 500,   # 16 templates → ~31 per template
    "direct_hard": 500,     # 12 templates → ~42 per template  
    "sequential": 500,      # 22 templates → ~23 per template
    # "parallel": 0,        # Skip parallel
}

# Templates per type
TEMPLATES = {
    "direct_simple": [
        "which_specialty", "what_risk_factor", "what_type", "mechanism", "how_regulate",
        "on_chromosome", "is_notifiable", "can_lead_to", "does_include", "in_what_year",
        "eponymous_disorder", "anatomy_affected", "how_many_count", "which_organization",
        "medication_for_symptom", "cytokine_role"
    ],
    "direct_hard": [
        "autoimmune_multi_symptom", "syndrome_multi_feature", "gene_dual_classification",
        "condition_complication_prone", "gene_dual_phenotype", "syndrome_gene_cause",
        "disorder_secondary_effect", "mutation_secondary_condition", "gene_both_conditions",
        "treatment_dual_condition", "cell_role_pathogenesis", "deficiency_associated_disorder"
    ],
    "sequential": [
        "chromosome_gene_disease",      # 47%
        "gene_mutation_condition",      # 11%
        "pathogen_disease_symptom",     # 5%
        "year_entity_discovery",        # 4%
        "eponym_syndrome",              # 7%
        "treatment_disease_symptom",    # 3%
        "inheritance_pattern_condition",
        "gene_property_count",
        "gene_family_member",
        "disease_symptom_combo",
    ],
}

CONFIG_PATH = Path(__file__).parent / "config.yml"
GENERATED_DIR = Path(__file__).parent / "output" / "generated"

def get_existing_samples(qtype: str, pattern: str) -> int:
    """Count samples already generated for this type/pattern combination."""
    total = 0
    # Look for files matching pattern: {qtype}_{pattern}_v*.json
    for f in GENERATED_DIR.glob(f"{qtype}_{pattern}_v*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                count = len(data.get("samples", []))
                total += count
        except:
            pass
    return total

def update_config(qtype: str, pattern: str, num_samples: int):
    """Update config.yml with new settings."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    
    config["generation"]["question_type"] = qtype
    config["generation"]["template_pattern"] = pattern
    config["generation"]["num_samples"] = num_samples
    config["generation"]["batch_size"] = min(30, num_samples)
    config["generation"]["reset_version"] = False  # Keep incrementing
    
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def run_generation():
    result = subprocess.run(
        ["python", "generate_samples.py"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        timeout=300
    )
    return result.returncode == 0, result.stdout, result.stderr

def main():
    print("=" * 70)
    print("BATCH GENERATION - Equal distribution per template (with RESUME)")
    print("=" * 70)
    
    # Scan existing generated files
    print("\n📊 Scanning existing generated files...")
    existing = {}
    total_existing = 0
    for qtype in GENERATIONS.keys():
        existing[qtype] = {}
        for pattern in TEMPLATES.get(qtype, []):
            count = get_existing_samples(qtype, pattern)
            existing[qtype][pattern] = count
            total_existing += count
    
    print(f"  Found {total_existing} existing samples\n")
    
    # Calculate what's needed
    total_planned = 0
    total_needed = 0
    for qtype, total in GENERATIONS.items():
        if total > 0:
            templates = TEMPLATES.get(qtype, [])
            per_template = total // len(templates)
            
            needed = 0
            for pattern in templates:
                have = existing[qtype].get(pattern, 0)
                need = max(0, per_template - have)
                needed += need
            
            print(f"{qtype}: {total} target / {len(templates)} templates = {per_template} per template")
            print(f"  → Already have: {sum(existing[qtype].values())}, Need: {needed}")
            total_planned += total
            total_needed += needed
    
    print(f"\n📌 Total planned: {total_planned}, Already have: {total_existing}, Need: {total_needed}")
    print("=" * 70)
    
    if total_needed == 0:
        print("\n✅ All templates already have enough samples!")
        return
    
    input("\nPress Enter to start generation (Ctrl+C to cancel)...")
    
    total_generated = 0
    skipped = 0
    failed = []
    
    for qtype, total_samples in GENERATIONS.items():
        if total_samples == 0:
            continue
            
        templates = TEMPLATES.get(qtype, [])
        samples_per_template = total_samples // len(templates)
        remainder = total_samples % len(templates)
        
        print(f"\n{'='*70}")
        print(f"GENERATING: {qtype.upper()}")
        print(f"{'='*70}")
        
        for i, pattern in enumerate(templates):
            # Distribute remainder across first few templates
            target = samples_per_template + (1 if i < remainder else 0)
            have = existing[qtype].get(pattern, 0)
            need = max(0, target - have)
            
            if need == 0:
                print(f"[{i+1}/{len(templates)}] {pattern}: ✓ already have {have}/{target}")
                skipped += 1
                continue
            
            print(f"\n[{i+1}/{len(templates)}] {pattern}: need {need} (have {have}/{target})")
            
            update_config(qtype, pattern, need)
            
            success, stdout, stderr = run_generation()
            
            if success:
                # Extract actual count from output
                for line in stdout.split("\n"):
                    if "Samples:" in line:
                        print(f"  {line.strip()}")
                        break
                total_generated += need
            else:
                print(f"  ✗ FAILED: {stderr[:200]}")
                failed.append(f"{qtype}/{pattern}")
            
            time.sleep(0.5)  # Rate limiting
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Generated: ~{total_generated}")
    print(f"Skipped (already done): {skipped}")
    if failed:
        print(f"Failed: {failed}")
    print("\nOutput files in: augment_data/output/generated/")

if __name__ == "__main__":
    main()
