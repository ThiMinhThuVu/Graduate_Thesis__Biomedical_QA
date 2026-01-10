"""
Synthetic Biomedical Question Generation Pipeline
==================================================

Combined pipeline with core logic and runner functionality.
Prompts are imported from data_generation_prompt.py

Two-Phase Approach:
- Phase 1: Generate question structure templates
- Phase 2: Fill templates with real biomedical entities

Usage:
    # As module
    from synthetic_data_generator import SyntheticPipeline
    pipeline = SyntheticPipeline()
    
    # As script
    python synthetic_data_generator.py
    python synthetic_data_generator.py --config custom_config.yml
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

# Import prompts from separate file
from data_generation_prompt import (
    DIRECT_DEFINITION,
    SEQUENTIAL_DEFINITION,
    PARALLEL_DEFINITION,
    BIOMEDICAL_CONTEXT,
    prompt_template_sequential_question,
    prompt_template_parallel_question,
    prompt_template_direct_question,
    prompt_question_generation,
)

# Try to import LLM client
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.gpt_request import request_api
    HAS_LLM = True
except ImportError:
    HAS_LLM = False


# =============================================================================
# TEMPLATE DEFINITIONS (Aligned with MedHopQA Analysis)
# =============================================================================

TEMPLATES = {
    "direct": [
        # --- COMMENTED: Similar to BioHopR/BioKGBench ---
        # {
        #     "pattern": "which_association",
        #     "structure": "Which [ENTITY?] is associated with [CONDITION] and can lead to [SYMPTOM]?",
        #     "wh_word": "Which",
        #     "topic": "Diagnosis"
        # },  # Similar to BioKGBench: "Which diseases are associated with {PROT}?"
        # {
        #     "pattern": "which_type",
        #     "structure": "Which type of [ENTITY] is associated with [CONDITION]?",
        #     "wh_word": "Which",
        #     "topic": "Diagnosis"
        # },  # Similar to BioHopR: "Name a Type({Query}) that is Label({Relationship}) by {Query}"
        # --- END COMMENTED ---
        
        {
            "pattern": "which_specialty",
            "structure": "Which medical specialty is most likely to treat patients with [DISEASE]?",
            "wh_word": "Which",
            "topic": "Treatment"
        },
        {
            "pattern": "what_risk_factor",
            "structure": "What is a common risk factor for both [DISEASE_A] and [DISEASE_B]?",
            "wh_word": "What",
            "topic": "Prevention"
        },
        {
            "pattern": "what_type_nevus",
            "structure": "What type of [ENTITY] is associated with [CONDITION]?",
            "wh_word": "What",
            "topic": "Diagnosis"
        },
        {
            "pattern": "mechanism",
            "structure": "How does [CHEMICAL] contribute to the development of [DISEASE]?",
            "wh_word": "How",
            "topic": "Pathophysiology"
        },
        {
            "pattern": "how_contribute",
            "structure": "How do [ENTITY_PLURAL] contribute to the regulation of [PROCESS]?",
            "wh_word": "How",
            "topic": "Pathophysiology"
        },
        {
            "pattern": "on_chromosome",
            "structure": "On which chromosome is the [GENE] gene, known to cause [DISEASE], located?",
            "wh_word": "On which",
            "topic": "Pathophysiology"
        },
        {
            "pattern": "in_year",
            "structure": "In which year was the treatment for [DISEASE] first approved?",
            "wh_word": "In which",
            "topic": "Treatment"
        },
        {
            "pattern": "is_notifiable",
            "structure": "Is [DISEASE] a notifiable disease?",
            "wh_word": "Is",
            "topic": "Prevention"
        },
        {
            "pattern": "can_lead_to",
            "structure": "Can [CONDITION] lead to [SYMPTOM]?",
            "wh_word": "Can",
            "topic": "Diagnosis"
        },
        {
            "pattern": "does_include",
            "structure": "Does [SYNDROME] include [SYMPTOM] as a symptom?",
            "wh_word": "Does",
            "topic": "Diagnosis"
        },
    ],
    
    "sequential": [
        # --- COMMENTED: Similar to BioHopR/BioKGBench ---
        # {
        #     "pattern": "which_gene_disease",
        #     "structure": "Which [GENE?] is mutated in the [DISEASE?] that causes [SYMPTOM]?",
        #     "chain": "[SYMPTOM] → [DISEASE] → [GENE]",
        #     "wh_word": "Which",
        #     "topic": "Pathophysiology"
        # },  # Similar to BioKGBench: "Which diseases associated with protein from gene"
        # {
        #     "pattern": "which_protein_encoded",
        #     "structure": "Which [PROTEIN?] is encoded by the [GENE?] that is mutated in [DISEASE]?",
        #     "chain": "[DISEASE] → [GENE] → [PROTEIN]",
        #     "wh_word": "Which",
        #     "topic": "Pathophysiology"
        # },  # Similar to BioKGBench: "molecular function of protein translated from gene"
        # --- END COMMENTED ---
        
        {
            "pattern": "what_treatment_caused_by",
            "structure": "What [DRUG?] is used to treat the [DISEASE?] caused by [GENE] mutations?",
            "chain": "[GENE] → [DISEASE] → [DRUG]",
            "wh_word": "What",
            "topic": "Treatment"
        },
        {
            "pattern": "what_pathway_disrupted",
            "structure": "What [PATHWAY?] is disrupted in the [DISEASE?] characterized by [SYMPTOM]?",
            "chain": "[SYMPTOM] → [DISEASE] → [PATHWAY]",
            "wh_word": "What",
            "topic": "Pathophysiology"
        },
        {
            "pattern": "on_chromosome_causes",
            "structure": "On which chromosome is the [GENE?] that causes [DISEASE?] characterized by [SYMPTOM]?",
            "chain": "[SYMPTOM] → [DISEASE] → [GENE] → [CHROMOSOME]",
            "wh_word": "On which",
            "topic": "Pathophysiology"
        },
        {
            "pattern": "is_same_gene",
            "structure": "Is the deficiency in [PROTEIN?] caused by the same gene that is mutated in [DISEASE]?",
            "chain": "[DISEASE] → [GENE] → [PROTEIN deficiency]",
            "wh_word": "Is",
            "topic": "Diagnosis"
        },
        {
            "pattern": "can_mutation_cause",
            "structure": "Can a mutation in the [GENE?] that encodes [PROTEIN] cause [DISEASE]?",
            "chain": "[PROTEIN] → [GENE] → [DISEASE]",
            "wh_word": "Can",
            "topic": "Pathophysiology"
        },
        {
            "pattern": "do_mutations_lead",
            "structure": "Do mutations in [GENE?] lead to [DISEASE?] that also causes [SYMPTOM]?",
            "chain": "[SYMPTOM] → [DISEASE] → [GENE]",
            "wh_word": "Do",
            "topic": "Pathophysiology"
        },
    ],
    
    "parallel": [
        {
            "pattern": "or_choice_gene",
            "structure": "Is [DISEASE] caused by mutations in [GENE_A] or [GENE_B]?",
            "branches": ["[DISEASE] → [GENE_A]?", "[DISEASE] → [GENE_B]?"],
            "conjunction": "or",
            "wh_word": "Is",
            "topic": "Pathophysiology"
        },
        {
            "pattern": "or_choice_treatment",
            "structure": "Which is more effective for treating [DISEASE], [DRUG_A] or [DRUG_B]?",
            "branches": ["[DISEASE] ← [DRUG_A]", "[DISEASE] ← [DRUG_B]"],
            "conjunction": "or",
            "wh_word": "Which",
            "topic": "Treatment"
        },
        
        # --- COMMENTED: Similar to BioKGBench ---
        # {
        #     "pattern": "associated_with",
        #     "structure": "Which [GENE?] is associated with [DISEASE] and involved in [PATHWAY]?",
        #     "branches": ["[DISEASE] → gene?", "[PATHWAY] → gene?"],
        #     "conjunction": "and",
        #     "wh_word": "Which",
        #     "topic": "Pathophysiology"
        # },  # Similar to BioKGBench: "Which pathways include both P1 and P2?"
        # {
        #     "pattern": "both_same_gene",
        #     "structure": "Are [DISEASE_A] and [DISEASE_B] both caused by mutations in the same [GENE?]?",
        #     "branches": ["[DISEASE_A] → gene?", "[DISEASE_B] → gene?"],
        #     "conjunction": "both",
        #     "wh_word": "Are",
        #     "topic": "Pathophysiology"
        # },  # Similar to BioKGBench: "Which diseases associated with both P1 and P2?"
        # {
        #     "pattern": "both_deficient",
        #     "structure": "Which [ENZYME?] is deficient in both [CONDITION_A] and [CONDITION_B]?",
        #     "branches": ["[CONDITION_A] → enzyme?", "[CONDITION_B] → enzyme?"],
        #     "conjunction": "both",
        #     "wh_word": "Which",
        #     "topic": "Diagnosis"
        # },  # Similar to BioKGBench: "Which molecular functions shared by P1 and P2?"
        # {
        #     "pattern": "same_mechanism",
        #     "structure": "Are the mechanisms causing [CONDITION_A] and [CONDITION_B] the same?",
        #     "branches": ["[CONDITION_A] → mechanism", "[CONDITION_B] → mechanism"],
        #     "conjunction": "same",
        #     "wh_word": "Are",
        #     "topic": "Pathophysiology"
        # },  # Similar to BioKGBench: shared functions pattern
        # --- END COMMENTED ---
        
        {
            "pattern": "difference",
            "structure": "What [SYMPTOM?] occurs in [DISEASE_A] but not in [DISEASE_B]?",
            "branches": ["[DISEASE_A] → symptoms", "[DISEASE_B] → symptoms"],
            "conjunction": "but not",
            "wh_word": "What",
            "topic": "Diagnosis"
        },
        {
            "pattern": "dual_chain_family",
            "structure": "Is the [GENE?] mutated in [DISEASE_A] from the same family as the [GENE?] in [DISEASE_B]?",
            "branches": ["[DISEASE_A] → gene → family", "[DISEASE_B] → gene → family"],
            "conjunction": "as",
            "wh_word": "Is",
            "topic": "Pathophysiology"
        },
    ]
}

# Default biomedical entities
DEFAULT_ENTITIES = {
    "diseases": [
        "Gaucher disease", "Fabry disease", "Parkinson's disease",
        "Huntington's disease", "Cystic fibrosis", "Phenylketonuria",
        "Duchenne muscular dystrophy", "Tay-Sachs disease"
    ],
    "genes": ["GBA", "GLA", "SNCA", "HTT", "CFTR", "PAH", "DMD", "HEXA"],
    "proteins": [
        "Glucocerebrosidase", "Alpha-galactosidase", "Alpha-synuclein",
        "Huntingtin", "CFTR", "Phenylalanine hydroxylase", "Dystrophin"
    ],
    "drugs": ["Imiglucerase", "Agalsidase", "Levodopa", "Ivacaftor", "Sapropterin"],
    "symptoms": ["Hepatomegaly", "Neuropathy", "Tremor", "Muscle weakness"],
    "pathways": ["Sphingolipid metabolism", "Dopamine signaling", "Glycolysis"]
}


# =============================================================================
# PIPELINE CLASS
# =============================================================================

class SyntheticPipeline:
    """Simple two-phase synthetic data generation pipeline."""
    
    def __init__(self):
        self.templates = TEMPLATES
        self.generated = []
        self.used_patterns = {}
        self.used_topics = {}
    
    # -------------------------------------------------------------------------
    # PHASE 1: Template Generation
    # -------------------------------------------------------------------------
    
    def get_template_prompt(self, question_type: str, num: int = 5) -> Dict[str, str]:
        """
        Phase 1: Get prompt to generate new templates.
        
        Args:
            question_type: "direct", "sequential", or "parallel"
            num: Number of templates to generate
        
        Returns:
            Dict with "system" and "user" prompts
        """
        patterns = {
            "direct": "[Wh-word] [unknown entity] [relation] [given attributes]",
            "sequential": "[Wh-word] [unknown A] [relation₁] [unknown B] [relation₂] [given C]",
            "parallel": "[Wh-word] [unknown A] [relation] [given B] [CONJUNCTION] [relation] [given C]"
        }
        
        system = """## Role
You are a biomedical question structure designer.

## Task
Generate question templates for biomedical Q&A. Use placeholders like [DISEASE], [GENE?] (? marks the answer).

## Constraints
- MUST: Use entity placeholders, not real names
- MUST: Mark unknown entities (answers) with "?"
- MUST: Vary Wh-words (What, Which, How, Is, Are, Do)"""

        user = f"""## Task
Generate {num} unique {question_type} question templates.

## Pattern
{patterns.get(question_type, patterns["direct"])}

## Entity Types
DISEASE, GENE, PROTEIN, ENZYME, DRUG, CHEMICAL, PATHWAY, SYMPTOM

## Examples from library
{json.dumps(self.templates.get(question_type, [])[:2], indent=2)}

## Output Format (JSON)
[
    {{
        "pattern": "descriptive_name",
        "structure": "Template with [ENTITY] placeholders",
        "wh_word": "What",
        "topic": "Pathophysiology|Treatment|Diagnosis|Prevention"
    }}
]

Generate {num} new templates:"""

        return {"system": system, "user": user}
    
    def add_templates(self, question_type: str, templates: List[Dict]):
        """Add new templates to the library."""
        if question_type not in self.templates:
            self.templates[question_type] = []
        self.templates[question_type].extend(templates)
    
    # -------------------------------------------------------------------------
    # PHASE 2: Question Generation  
    # -------------------------------------------------------------------------
    
    def get_question_prompt(
        self,
        template: Optional[Dict] = None,
        question_type: Optional[str] = None,
        entities: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Phase 2: Get prompt to generate a question from a template.
        
        Args:
            template: Specific template to use (optional)
            question_type: Type if no template given
            entities: Available biomedical entities
        
        Returns:
            Dict with "system" and "user" prompts
        """
        # Select template
        if template is None:
            template = self._select_template(question_type)
        
        # Use default entities if not provided
        if entities is None:
            entities = DEFAULT_ENTITIES
        
        # Get diversity hints
        hints = self._get_diversity_hints()
        hints_str = "\n".join(f"- {h}" for h in hints) if hints else "None"
        
        system = """## Role
You are a biomedical question generator for MedHopQA dataset.

## Task
Fill the template with real biomedical entities to create a factual question.

## Constraints
- MUST: Use only factually accurate information
- MUST: Ensure exactly ONE correct answer
- MUST: Short answer must be 1-3 words maximum
- MUST NOT: Create hypothetical questions"""

        user = f"""## Template
**Pattern:** {template.get('pattern', 'unknown')}
**Structure:** {template['structure']}
**Topic:** {template.get('topic', 'Pathophysiology')}

## Available Entities
{json.dumps(entities, indent=2)}

## Diversity Hints
{hints_str}

## Output Format (JSON)
{{
    "question": "The complete question with real entities",
    "short_answer": "1-3 word answer",
    "reasoning": "Step-by-step reasoning chain",
    "topic": "{template.get('topic', 'Pathophysiology')}",
    "entities_used": ["entity1", "entity2"]
}}

Generate a factually accurate question:"""

        return {"system": system, "user": user}
    
    def _select_template(self, question_type: Optional[str] = None) -> Dict:
        """Select a template, preferring less-used patterns."""
        # Get candidates
        if question_type:
            candidates = self.templates.get(question_type, [])
        else:
            # Mix all types
            candidates = []
            for templates in self.templates.values():
                candidates.extend(templates)
        
        if not candidates:
            raise ValueError("No templates available")
        
        # Score by inverse usage (prefer less used)
        scored = []
        for t in candidates:
            pattern = t.get("pattern", "")
            score = -self.used_patterns.get(pattern, 0) + random.random()
            scored.append((score, t))
        
        # Return best
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
    
    def _get_diversity_hints(self) -> List[str]:
        """Generate hints to increase diversity."""
        hints = []
        
        # Suggest underused topics
        all_topics = ["Pathophysiology", "Treatment", "Diagnosis", "Prevention"]
        counts = [self.used_topics.get(t, 0) for t in all_topics]
        if counts:
            min_topic = all_topics[counts.index(min(counts))]
            hints.append(f"Consider topic: {min_topic}")
        
        return hints
    
    def record(self, template: Dict, result: Dict):
        """Record a generation for diversity tracking."""
        pattern = template.get("pattern", "")
        topic = result.get("topic", template.get("topic", ""))
        
        self.used_patterns[pattern] = self.used_patterns.get(pattern, 0) + 1
        self.used_topics[topic] = self.used_topics.get(topic, 0) + 1
        self.generated.append(result)
    
    # -------------------------------------------------------------------------
    # Batch Generation
    # -------------------------------------------------------------------------
    
    def generate_batch_prompts(
        self,
        num: int,
        question_type: Optional[str] = None,
        entities: Optional[Dict] = None
    ) -> List[Dict]:
        """Generate prompts for a batch of questions."""
        prompts = []
        
        for i in range(num):
            # Rotate types if not specified
            if question_type is None:
                types = list(self.templates.keys())
                qtype = types[i % len(types)]
            else:
                qtype = question_type
            
            template = self._select_template(qtype)
            prompt = self.get_question_prompt(template, entities=entities)
            prompts.append({
                "index": i,
                "type": qtype,
                "template": template,
                "prompt": prompt
            })
        
        return prompts
    
    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    
    def export(self, filepath: str):
        """Export generated questions to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.generated, f, indent=2)
    
    def stats(self) -> Dict:
        """Get generation statistics."""
        return {
            "total_templates": sum(len(t) for t in self.templates.values()),
            "total_generated": len(self.generated),
            "pattern_usage": dict(self.used_patterns),
            "topic_usage": dict(self.used_topics)
        }


# =============================================================================
# QUICK FUNCTIONS
# =============================================================================

def phase1_prompt(question_type: str, num: int = 5) -> Dict[str, str]:
    """Quick function to get Phase 1 prompt."""
    return SyntheticPipeline().get_template_prompt(question_type, num)


def phase2_prompt(
    template: Optional[Dict] = None,
    question_type: str = "sequential",
    entities: Optional[Dict] = None
) -> Dict[str, str]:
    """Quick function to get Phase 2 prompt."""
    pipeline = SyntheticPipeline()
    if template is None:
        template = pipeline._select_template(question_type)
    return pipeline.get_question_prompt(template, entities=entities)


# =============================================================================
# RUNNER FUNCTIONS
# =============================================================================

def load_config(config_path: str = "config.yml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent / config_path
    
    # Check command line for custom config
    if len(sys.argv) > 2 and sys.argv[1] == "--config":
        config_file = Path(sys.argv[2])
    
    if not config_file.exists():
        print(f"Config file not found: {config_file}")
        sys.exit(1)
    
    with open(config_file) as f:
        return yaml.safe_load(f)


def call_llm(prompt: dict, config: dict) -> str:
    """Call LLM with prompt."""
    if not HAS_LLM:
        return ""
    
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]}
    ]
    
    model = config.get("llm", {}).get("model", "gpt-4o-mini")
    response = request_api(messages, model=model)
    return response.get("content", str(response)) if isinstance(response, dict) else str(response)


def run_demo(pipeline: SyntheticPipeline):
    """Run demo mode - show example prompts."""
    print("=" * 50)
    print("DEMO MODE")
    print("=" * 50)
    
    print("\n--- Phase 1: Template Generation Prompt ---")
    p1 = pipeline.get_template_prompt("sequential", 3)
    print(f"System:\n{p1['system']}\n")
    print(f"User:\n{p1['user']}")
    
    print("\n--- Phase 2: Question Generation Prompt ---")
    p2 = pipeline.get_question_prompt(question_type="parallel")
    print(f"System:\n{p2['system']}\n")
    print(f"User:\n{p2['user']}")


def run_phase1(pipeline: SyntheticPipeline, config: dict):
    """Run Phase 1: Template Generation."""
    cfg = config.get("phase1", {})
    question_type = cfg.get("question_type", "sequential")
    num = cfg.get("num_templates", 5)
    output = cfg.get("output", "output/templates.json")
    
    print("=" * 50)
    print(f"PHASE 1: Generating {num} {question_type} templates")
    print("=" * 50)
    
    prompt = pipeline.get_template_prompt(question_type, num)
    
    if HAS_LLM:
        print("Calling LLM...")
        response = call_llm(prompt, config)
        
        # Parse and save
        try:
            # Extract JSON array from response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                templates = json.loads(response[start:end])
                pipeline.add_templates(question_type, templates)
                
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                with open(output, 'w') as f:
                    json.dump(templates, f, indent=2)
                
                print(f"✓ Generated {len(templates)} templates")
                print(f"✓ Saved to: {output}")
            else:
                print("Failed to parse templates from response")
                print(f"Response:\n{response}")
        except Exception as e:
            print(f"Error: {e}")
            print(f"Response:\n{response}")
    else:
        print("No LLM available. Saving prompt instead...")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(prompt, f, indent=2)
        print(f"Prompt saved to: {output}")


def run_phase2(pipeline: SyntheticPipeline, config: dict):
    """Run Phase 2: Question Generation."""
    cfg = config.get("phase2", {})
    question_type = cfg.get("question_type")  # None means mixed
    num = cfg.get("num_questions", 10)
    output = cfg.get("output", "output/questions.json")
    balance_types = cfg.get("balance_types", True)
    
    # Load entities from config
    entities = config.get("entities", None)
    
    print("=" * 50)
    print(f"PHASE 2: Generating {num} questions")
    print(f"Type: {question_type or 'mixed'}")
    print("=" * 50)
    
    results = []
    
    # Generate prompts
    batch = pipeline.generate_batch_prompts(
        num=num,
        question_type=question_type if not balance_types else None,
        entities=entities
    )
    
    for item in batch:
        idx = item['index'] + 1
        print(f"[{idx}/{num}] {item['type']}: {item['template']['pattern']}", end=" ")
        
        if HAS_LLM:
            response = call_llm(item['prompt'], config)
            
            try:
                # Extract JSON from response
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    result = json.loads(response[start:end])
                    pipeline.record(item['template'], result)
                    results.append(result)
                    print("✓")
                else:
                    print("✗ (no JSON)")
            except Exception as e:
                print(f"✗ ({e})")
        else:
            print("(no LLM)")
    
    # Save results
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    if results:
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved {len(results)} questions to: {output}")
    else:
        # Save prompts instead
        prompts = [{"template": p["template"], "prompt": p["prompt"]} for p in batch]
        with open(output, 'w') as f:
            json.dump(prompts, f, indent=2)
        print(f"\n✓ Saved {len(prompts)} prompts to: {output}")
    
    print(f"\nStats: {pipeline.stats()}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the synthetic data generator."""
    # Load config
    config = load_config()
    mode = config.get("mode", "demo")
    
    print(f"Config loaded. Mode: {mode}")
    if not HAS_LLM:
        print("Note: LLM not available. Running in prompt-only mode.")
    
    # Initialize pipeline
    pipeline = SyntheticPipeline()
    
    # Run based on mode
    if mode == "demo":
        run_demo(pipeline)
    elif mode == "phase1":
        run_phase1(pipeline, config)
    elif mode == "phase2":
        run_phase2(pipeline, config)
    else:
        print(f"Unknown mode: {mode}")
        print("Valid modes: demo, phase1, phase2")


if __name__ == "__main__":
    main()
