# =============================================================================
# Biomedical Multi-Hop Question Data Generation Prompts
# Aligned with MedHopQA dataset characteristics
# Focus: Diseases, Genes/Proteins, Chemicals
# Topics: Pathophysiology, Treatment, Diagnosis, Prevention
# =============================================================================

from typing import Dict, List, Optional

# =============================================================================
# QUESTION TYPE DEFINITIONS
# =============================================================================

QUESTION_TYPE_DEFINITIONS = {
    "direct": """## Direct Question Type (d)

### Definition
The question can be reasoned about immediately as a single unit. It involves tightly interdependent parts that cannot be separated without losing the core intent.

### Key Characteristic
Describes a single disease, gene, protein, or chemical entity with multiple attributes or requires a direct biomedical lookup/reasoning.

### Question Structure Templates

Use these templates to create diverse direct questions. The answer requires single-step reasoning.

**Template Pattern:**
```
[Wh-word] [unknown entity] [relation] [given entity/attributes]
```

**Structural Variations:**

1. **Entity-Attribute Query**
   - "What [ENTITY] is characterized by [ATTRIBUTE_1] and [ATTRIBUTE_2]?"
   - "Which [DISEASE] presents with [SYMPTOM_A], [SYMPTOM_B], and [SYMPTOM_C]?"

2. **Mechanism Query**
   - "How does [ENTITY_A] contribute to [PROCESS/CONDITION]?"
   - "What is the mechanism by which [DRUG] treats [DISEASE]?"

3. **Definition/Function Query**
   - "What is the primary function of [PROTEIN/GENE]?"
   - "Which [ENZYME] catalyzes [REACTION] in [PATHWAY]?"

4. **Association Query**
   - "What [ENTITY] is associated with [CONDITION] and [FEATURE]?"
   - "Which [GENE] is linked to [DISEASE] through [MECHANISM]?"

**Wh-word Diversity:**
- What, Which, Who (for entity identification)
- How (for mechanisms)
- Why (for causation)

### Examples

**Example 1:** "What enzyme is deficient in phenylketonuria, leading to phenylalanine accumulation?"
- Structure: [ENZYME?] ← deficient in ← [PKU] + [causes phenylalanine accumulation]
- Answer: Phenylalanine hydroxylase

**Example 2:** "Which autoimmune disease is associated with chronic urticaria and causes dry mouth and dry eyes?"
- Structure: [DISEASE?] ← has features ← [chronic urticaria, dry mouth, dry eyes]
- Answer: Sjögren's syndrome

**Example 3:** "How does HMG-CoA reductase inhibition lower cholesterol levels?"
- Structure: [MECHANISM?] ← [HMG-CoA reductase inhibition] → [cholesterol lowering]
- Answer: Blocks cholesterol synthesis pathway
""",

    "sequential": """## Sequential Question Type (s)

### Definition
The question contains sub-questions that depend on each other. You must answer the first part to identify the subject of the second part. This reflects the multi-hop nature of biomedical reasoning.

### Key Characteristic
Step-by-step biomedical filtering (Disease → Gene → Protein → Answer or Chemical → Pathway → Effect).

### Question Structure Templates

Use these templates to create diverse sequential questions. The answer requires resolving entities in order.

**Template Pattern:**
```
[Wh-word] [unknown entity A] [relation₁] [unknown entity B] [relation₂] [given entity C]
```

**Structural Variations:**

1. **Disease → Gene → Answer**
   - "What [GENE] is mutated in the [DISEASE] that causes [SYMPTOM]?"
   - "Which [GENE] encodes the protein deficient in [DISEASE] characterized by [FEATURE]?"

2. **Gene → Protein → Answer**
   - "What [PROTEIN] is encoded by the [GENE] that is mutated in patients with [DISEASE]?"
   - "Which [ENZYME] is produced by the [GENE] associated with [CONDITION]?"

3. **Chemical → Pathway → Answer**
   - "What [DRUG] targets the [PATHWAY] disrupted in [DISEASE]?"
   - "Which [CHEMICAL] inhibits the [ENZYME] that causes [CONDITION]?"

4. **Disease → Treatment → Answer**
   - "What [DRUG] is used to treat the [DISEASE] caused by [GENE] mutations?"
   - "Which [THERAPY] addresses the [DEFICIENCY] seen in [DISEASE]?"

5. **Symptom → Disease → Answer**
   - "What [GENE] causes the [DISEASE] that presents with [SYMPTOM] and [SYMPTOM]?"
   - "Which [PROTEIN] is deficient in the [CONDITION] leading to [MANIFESTATION]?"

**Wh-word Diversity:**
- What, Which, Who (for researchers/discoverers)
- Is/Are, Does/Do, Can (for yes/no chains)
- How (for mechanism chains)

### Examples

**Example 1:** "Which gene is mutated in the lysosomal storage disorder that is treated with Imiglucerase?"
- Structure: [GENE?] ← mutated in ← [DISEASE?] ← treated with ← [Imiglucerase]
- Chain: Imiglucerase → Gaucher disease → GBA gene

**Example 2:** "What protein accumulates in the brains of patients with the disease caused by SNCA mutations?"
- Structure: [PROTEIN?] ← accumulates in ← [DISEASE?] ← caused by ← [SNCA]
- Chain: SNCA → Parkinson's disease → Alpha-synuclein

**Example 3:** "What drug targets the fusion protein created by the chromosomal translocation in chronic myeloid leukemia?"
- Structure: [DRUG?] ← targets ← [PROTEIN?] ← created by ← [CML translocation]
- Chain: CML → BCR-ABL → Imatinib
""",

    "parallel": """## Parallel Question Type (p)

### Definition
The question contains independent sub-questions that can be queried simultaneously. The final answer is synthesized by comparing or finding intersections between biomedical entities.

### Key Characteristic
Independent branches (Gene_A and Gene_B are queried separately, then compared).

### Question Structure Templates

Use these templates to create diverse parallel questions. Each branch can be resolved independently.

---

**NORMAL Pattern (Two-Branch Comparison):**
```
[Wh-word] [unknown entity A] [relation] [given entity B] [CONJUNCTION] [relation] [given entity C]
```

**Structural Variations - Normal:**

1. **Same Property Comparison (Are/Do both...)**
   - "Are [DISEASE_A] and [DISEASE_B] both caused by mutations in [GENE_FAMILY]?"
   - "Do both [DRUG_A] and [DRUG_B] target the same [RECEPTOR/PATHWAY]?"
   - "Can [CONDITION_A] and [CONDITION_B] both lead to [SYMPTOM]?"

2. **Shared Entity Query (Which...both)**
   - "Which [GENE] is associated with both [DISEASE_A] and [DISEASE_B]?"
   - "What [ENZYME] is deficient in both [CONDITION_A] and [CONDITION_B]?"
   - "Which [DRUG] can treat both [DISEASE_A] and [DISEASE_B]?"

3. **Difference Query (but not/while)**
   - "What [GENE] is mutated in [DISEASE_A] but not in [DISEASE_B]?"
   - "Which [SYMPTOM] occurs in [CONDITION_A] but is absent in [CONDITION_B]?"

4. **Comparison Query (Which...more/earlier/different)**
   - "Which disease, [DISEASE_A] or [DISEASE_B], has an earlier onset?"
   - "Is [DRUG_A] or [DRUG_B] more effective for treating [CONDITION]?"

---

**COMPLEX Pattern (Multi-Branch Comparison):**
```
[Wh-word] [unknown A] [relation] [unknown B] [relation] [given C] [CONJUNCTION] [relation] [unknown D] [relation] [given E]
```

**Structural Variations - Complex:**

1. **Dual Chain Comparison**
   - "Is the [GENE] mutated in [DISEASE_A] from the same family as the [GENE] mutated in [DISEASE_B]?"
   - "Does the [PROTEIN] deficient in [CONDITION_A] share a pathway with the [PROTEIN] deficient in [CONDITION_B]?"

2. **Cross-Entity Comparison**
   - "Are the [DRUGS] used to treat [DISEASE_A] and [DISEASE_B] both targeting the same [MECHANISM]?"
   - "Do the [GENES] causing [SYNDROME_A] and [SYNDROME_B] encode proteins in the same [PATHWAY]?"

3. **Triple Branch (Intersection)**
   - "Which [GENE_FAMILY] includes genes mutated in [DISEASE_A], [DISEASE_B], and [DISEASE_C]?"
   - "What [PATHWAY] is disrupted by [CHEMICAL_A], [CHEMICAL_B], or [CHEMICAL_C]?"

---

**Conjunction Word Diversity:**
- Comparison: and, both...and, as well as
- Contrast: but, however, while, whereas, but not
- Alternative: or, either...or
- Additive: also, in addition to, along with

**Wh-word Diversity:**
- What, Which (for entity queries)
- Are, Do, Does, Can, Is (for yes/no comparisons)
- How (for mechanism comparisons)

### Examples

**Example 1 (Normal - Same Property):** "Are both Duchenne and Becker muscular dystrophy caused by mutations in the same gene?"
- Branch 1: Gene for Duchenne? → DMD gene
- Branch 2: Gene for Becker? → DMD gene
- Synthesis: Compare → Yes, same gene

**Example 2 (Normal - Shared Entity):** "Which enzyme is deficient in both Niemann-Pick type A and type B?"
- Branch 1: Enzyme in NP-A? → Sphingomyelinase
- Branch 2: Enzyme in NP-B? → Sphingomyelinase
- Synthesis: Intersection → Sphingomyelinase

**Example 3 (Normal - Contrast):** "Do metformin and sulfonylureas both increase insulin secretion?"
- Branch 1: Metformin mechanism? → No, increases sensitivity
- Branch 2: Sulfonylureas mechanism? → Yes, increases secretion
- Synthesis: Compare → No, only sulfonylureas

**Example 4 (Complex - Dual Chain):** "Is the gene mutated in Gaucher disease from the same family as the gene mutated in Fabry disease?"
- Chain 1: Gaucher → GBA gene → Glycosidase family
- Chain 2: Fabry → GLA gene → Glycosidase family
- Synthesis: Compare families → Yes, both glycosidases

**Example 5 (Complex - Cross-Entity):** "Do the drugs targeting BCR-ABL in CML and EGFR in lung cancer share the same mechanism of action?"
- Chain 1: CML → BCR-ABL → Imatinib → Tyrosine kinase inhibitor
- Chain 2: Lung cancer → EGFR → Erlotinib → Tyrosine kinase inhibitor
- Synthesis: Compare mechanisms → Yes, both TKIs
"""
}

# =============================================================================
# BIOMEDICAL CONTEXT (Shared across all prompts)
# =============================================================================

BIOMEDICAL_CONTEXT = """## Domain Focus
Biomedicine and Rare Diseases

### Entity Types
- Diseases
- Genes/Proteins
- Chemicals/Drugs

### Medical Categories
- Pathophysiology and Biology: Molecular mechanisms, disease etiology, biological pathways
- Medical Treatment: Drug therapies, enzyme replacement, gene therapy
- Diagnosis and Prognosis: Biomarkers, genetic testing, disease progression
- Disease Prevention: Prophylactic measures, lifestyle interventions, screening

## Constraints
- MUST: Questions must relate to treatment, diagnosis, biology, or prevention
- MUST: Based on established medical facts, NOT hypothetical scenarios
- MUST: Information should be well-established and unlikely to change
- MUST: Answer requires synthesizing information from multiple sources/steps
- MUST: Question has only ONE correct conceptual answer
- MUST: Short answer is 1-3 words maximum (e.g., "Huntingtin protein", "CFTR gene", "Imatinib")

## Self-Validation Checklist
1. Is the question medically relevant?
2. Is the answer factually correct and well-established?
3. Does answering require multiple reasoning steps?
4. Is there only one correct answer?
5. Is the short answer 3 words or fewer?
6. Are accepted synonyms provided for the answer?
"""

# =============================================================================
# FEW-SHOT EXAMPLES (Organized by type)
# =============================================================================

FEW_SHOT_EXAMPLES = {
    "direct": [
        # From dev set: Q1 - Direct (given attributes → disorder)
        {
            "question": "What genetic disorder associated with hypoparathyroidism, sensorineural deafness, and renal disease can cause proteinuria?",
            "short_answer": "HDR syndrome",
            "reasoning": "HDR syndrome (Hypoparathyroidism, Deafness, Renal dysplasia) is characterized by hypoparathyroidism, sensorineural deafness, and renal abnormalities including proteinuria.",
            "topic": "Diagnosis"
        },
        # From dev set: Q2 - Direct (ranking)
        {
            "question": "What is the second most important cause of dementia after Alzheimer's disease?",
            "short_answer": "Vascular dementia",
            "reasoning": "Vascular dementia is the second most common cause of dementia, accounting for about 15-20% of cases, after Alzheimer's disease which accounts for 60-70%.",
            "topic": "Diagnosis"
        },
        # From dev set: Q30 - Direct (disease → specialist)
        {
            "question": "Which medical specialist would manage a patient with Dyschromatosis universalis hereditaria?",
            "short_answer": "Dermatologist",
            "reasoning": "Dyschromatosis universalis hereditaria is a rare genetic skin disorder characterized by hyperpigmented and hypopigmented macules, and would be managed by a dermatologist.",
            "topic": "Treatment"
        }
    ],
    
    "sequential": [
        # From dev set: Q3 - Sequential (function→enzyme→gene→chromosome)
        {
            "question": "Which human chromosome contains the gene that encodes the enzyme that can inactivate antithrombin?",
            "short_answer": "Chromosome 11",
            "reasoning": "Step 1: The enzyme that inactivates antithrombin is thrombin/Factor IIa. Step 2: Thrombin is encoded by the F2 (prothrombin) gene. Step 3: The F2 gene is located on chromosome 11q11-q12.",
            "topic": "Pathophysiology"
        },
        # From dev set: Q4 - Sequential (symptoms→syndrome→gene)
        {
            "question": "Which gene mutation is responsible for the syndrome characterized by flattened and deformed vertebrae, developmental delay, and lens coloboma?",
            "short_answer": "GNPAT gene",
            "reasoning": "Step 1: The syndrome with vertebral anomalies, developmental delay, and lens coloboma is rhizomelic chondrodysplasia punctata. Step 2: RCDP can be caused by mutations in several genes including GNPAT.",
            "topic": "Pathophysiology"
        },
        # From dev set: Q10 - Sequential (disease→gene→chromosome)
        {
            "question": "On which human chromosome is the gene that causes DFNB1 located?",
            "short_answer": "Chromosome 13",
            "reasoning": "Step 1: DFNB1 is autosomal recessive nonsyndromic hearing loss. Step 2: It is caused by mutations in GJB2 (connexin 26). Step 3: GJB2 is located on chromosome 13q12.",
            "topic": "Pathophysiology"
        }
    ],
    
    "parallel": [
        # From dev set: Q21 - Parallel (two conditions, shared symptom)
        {
            "question": "What is a common symptom shared by thyroid hormone resistance and goitre?",
            "short_answer": "Thyroid enlargement",
            "reasoning": "Branch 1: Thyroid hormone resistance features - includes compensatory thyroid enlargement. Branch 2: Goitre features - by definition involves thyroid enlargement. Synthesis: Both conditions share thyroid gland enlargement as a common feature.",
            "topic": "Diagnosis"
        },
        # From dev set: Q23 - Parallel (gene for both diseases)
        {
            "question": "Which keratin gene mutation is associated with both Pachyonychia congenita and Steatocystoma multiplex?",
            "short_answer": "KRT17",
            "reasoning": "Branch 1: Genes for Pachyonychia congenita - KRT6A, KRT6B, KRT16, KRT17. Branch 2: Genes for Steatocystoma multiplex - KRT17. Synthesis: KRT17 mutations can cause both conditions.",
            "topic": "Pathophysiology"
        },
        # From dev set: Q37 - Parallel (two diseases, shared symptom)
        {
            "question": "What is a common symptom shared by essential thrombocythemia and anemia?",
            "short_answer": "Fatigue",
            "reasoning": "Branch 1: Essential thrombocythemia symptoms - fatigue, weakness, headaches, bleeding. Branch 2: Anemia symptoms - fatigue, weakness, pallor, shortness of breath. Synthesis: Fatigue is a common symptom in both conditions.",
            "topic": "Diagnosis"
        }
    ]
}

# =============================================================================
# TYPE-SPECIFIC METADATA
# =============================================================================

TYPE_METADATA = {
    "direct": {
        "description": "involves direct reasoning about a single entity or tightly coupled relationship",
        "focus_areas": [
            "Disease mechanisms and etiology",
            "Gene/protein functions and associations",
            "Drug mechanisms and therapeutic effects",
            "Diagnostic markers and their significance"
        ]
    },
    "sequential": {
        "description": "requires step-by-step reasoning where the answer to one question is needed to answer the next",
        "focus_areas": [
            "Rare diseases and their genetic/molecular basis",
            "Gene-disease-protein relationships",
            "Drug mechanisms and therapeutic targets",
            "Pathophysiological pathways and disease progression"
        ]
    },
    "parallel": {
        "description": "requires investigating multiple independent facts and synthesizing them to form the answer",
        "focus_areas": [
            "Comparing genetic causes of different diseases",
            "Contrasting treatment mechanisms",
            "Finding shared pathways or molecular targets",
            "Identifying common or distinct biochemical features"
        ]
    }
}


# =============================================================================
# UNIFIED PROMPT GENERATION FUNCTION
# =============================================================================

def prompt_template_question(question_type: str) -> Dict:
    """
    Generate prompts for biomedical question generation based on question type.
    
    Args:
        question_type: One of 'direct', 'sequential', or 'parallel'
    
    Returns:
        Dict with 'system', 'user' prompts and 'few_shots' examples
    
    Raises:
        ValueError: If question_type is not valid
    """
    if question_type not in QUESTION_TYPE_DEFINITIONS:
        raise ValueError(f"Invalid question type: {question_type}. Must be one of: {list(QUESTION_TYPE_DEFINITIONS.keys())}")
    
    # Get type-specific content
    definition = QUESTION_TYPE_DEFINITIONS[question_type]
    metadata = TYPE_METADATA[question_type]
    few_shots = FEW_SHOT_EXAMPLES[question_type]
    
    # Build focus areas string
    focus_areas = "\n".join(f"- {area}" for area in metadata["focus_areas"])
    
    # Build system prompt
    system_prompt = f"""## Role
You are an expert biomedical data generator specializing in multi-hop question answering tasks for diseases, genes, proteins, and chemicals.

## Task
Create complex questions that require {'multi-hop ' if question_type != 'direct' else ''}reasoning over biomedical information, similar to the MedHopQA dataset.

## Focus Areas
{focus_areas}"""

    # Build user prompt
    user_prompt = f"""## Task
Generate a {question_type} {'multi-hop ' if question_type != 'direct' else ''}biomedical question.

## Instructions
1. The question MUST involve biomedical entities: diseases (especially rare diseases), genes, proteins, or chemicals/drugs
2. {'The question should focus on a single biomedical entity or tightly coupled relationship' if question_type == 'direct' else 'The question MUST require at least two reasoning steps to answer'}
3. {'It involves direct reasoning without requiring intermediate steps' if question_type == 'direct' else 'Both branches/steps must be needed to synthesize the final answer' if question_type == 'parallel' else 'The answer to the first step must be needed to answer the second step'}
4. Ensure the question is medically accurate and based on established facts
5. The short answer should be 1-3 words maximum
6. Focus on one of these topics: Pathophysiology, Treatment, Diagnosis, or Prevention
7. The question should be answerable using biomedical literature (e.g., PubMed, OMIM, DrugBank)

{BIOMEDICAL_CONTEXT}

{definition}"""

    return {
        "system": system_prompt,
        "user": user_prompt,
        "few_shots": few_shots
    }


def prompt_question_generation(template: str, question_type: str) -> Dict:
    """
    Generate a multi-hop biomedical question based on a given template and type.
    
    Args:
        template: The template structure for the question (e.g., MeSH term relationships)
        question_type: 'sequential', 'parallel', or 'direct'
    
    Returns:
        Aggregated prompt dictionary with system, user prompts and few-shot examples
    
    Raises:
        ValueError: If question_type is not valid
    """
    if question_type not in QUESTION_TYPE_DEFINITIONS:
        raise ValueError(f"Invalid question type: {question_type}. Must be one of: {list(QUESTION_TYPE_DEFINITIONS.keys())}")
    
    # Get type-specific content
    definition = QUESTION_TYPE_DEFINITIONS[question_type]
    metadata = TYPE_METADATA[question_type]
    few_shots = FEW_SHOT_EXAMPLES[question_type]
    
    system_prompt = """## Role
You are an expert biomedical data generator specializing in multi-hop question answering tasks.

## Task
Create complex questions about diseases, genes, proteins, and chemicals that require reasoning over multiple pieces of biomedical information, following the MedHopQA dataset format.

## Key Requirements
- Focus on rare diseases and their molecular/genetic basis
- Questions should be medically accurate and fact-based
- Answers should be 1-3 words maximum
- Include reasoning chains showing the multi-hop logic
- Cover topics: Pathophysiology, Treatment, Diagnosis, Prevention"""
    
    user_prompt = f"""## Task
Generate a {question_type} multi-hop biomedical question based on the following template and criteria.

## Template/Entity Relationships
{template}

## Question Type
{question_type} - This type {metadata['description']}.

## Instructions
1. The question MUST involve biomedical entities: diseases (especially rare diseases), genes/proteins, or chemicals/drugs
2. Ensure the question is medically accurate and based on established biomedical facts
3. The short answer should be 1-3 words maximum
4. The question must require multi-hop reasoning to answer
5. Focus on one of these topics: Pathophysiology, Treatment, Diagnosis, or Prevention

{BIOMEDICAL_CONTEXT}

## Definition of {question_type} Questions
{definition}

## Output Format (JSON)
Provide output as a valid JSON object with these fields:
{{
    "question": "The multi-hop biomedical question",
    "short_answer": "1-3 word answer",
    "accepted_synonyms": ["list", "of", "alternative", "valid", "answers"],
    "reasoning": "Step-by-step reasoning showing the multi-hop logic",
    "topic": "One of: Pathophysiology, Treatment, Diagnosis, Prevention",
    "entities": ["list of biomedical entities involved"],
    "validation": {{
        "is_medically_relevant": true/false,
        "is_factual": true/false,
        "is_stable": true/false,
        "requires_multihop": true/false,
        "has_unique_answer": true/false,
        "answer_format_valid": true/false
    }}
}}

## Constraints
- MUST: Only output questions where ALL validation fields are true"""
    
    return {
        "system": system_prompt,
        "user": user_prompt,
        "few_shots": few_shots
    }


# =============================================================================
# CONVENIENCE ALIASES (Backward Compatibility)
# =============================================================================

def prompt_template_sequential_question() -> Dict:
    """Generate prompts for sequential multi-hop biomedical questions."""
    return prompt_template_question("sequential")


def prompt_template_parallel_question() -> Dict:
    """Generate prompts for parallel multi-hop biomedical questions."""
    return prompt_template_question("parallel")


def prompt_template_direct_question() -> Dict:
    """Generate prompts for direct biomedical questions."""
    return prompt_template_question("direct")


# =============================================================================
# EXPORTED CONSTANTS (For external use)
# =============================================================================

# Keep these for backward compatibility
DIRECT_DEFINITION = QUESTION_TYPE_DEFINITIONS["direct"]
SEQUENTIAL_DEFINITION = QUESTION_TYPE_DEFINITIONS["sequential"]
PARALLEL_DEFINITION = QUESTION_TYPE_DEFINITIONS["parallel"]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("Unified Prompt Generation Demo")
    print("=" * 60)
    
    for qtype in ["direct", "sequential", "parallel"]:
        print(f"\n--- {qtype.upper()} Question Prompt ---")
        prompt = prompt_template_question(qtype)
        print(f"System prompt: {len(prompt['system'])} chars")
        print(f"User prompt: {len(prompt['user'])} chars")
        print(f"Few-shots: {len(prompt['few_shots'])} examples")
        print(f"First example: {prompt['few_shots'][0]['question'][:60]}...")
    
    print("\n--- Template-Based Generation ---")
    template = "Disease: Gaucher disease, Gene: GBA, Protein: Glucocerebrosidase"
    prompt = prompt_question_generation(template, "sequential")
    print(f"System prompt: {len(prompt['system'])} chars")
    print(f"User prompt preview:\n{prompt['user'][:300]}...")