from typing import Optional, List, Dict, Any
import json

# BIO_LABELS_LIST = ["DISEASE", "CHEMICAL", "DRUG", "GENE", "BODY_PART", "ORG", "GPE"]
BIO_LABELS_LIST = ["DNA", "CELL_TYPE", "CELL_LINE", "RNA", "PROTEIN", "DISEASE", "CHEMICAL", "AMINO_ACID", "ANATOMICAL_SYSTEM", "CANCER", "CELL", "CELLULAR_COMPONENT", "DEVELOPING_ANATOMICAL_STRUCTURE", "GENE_OR_GENE_PRODUCT", "IMMATERIAL_ANATOMICAL_ENTITY", "MULTI-TISSUE_STRUCTURE", "ORGAN", "ORGANISM", "ORGANISM_SUBDIVISION", "ORGANISM_SUBSTANCE", "PATHOLOGICAL_FORMATION", "SIMPLE_CHEMICAL", "TISSUE", "ORG", "GPE"]
WIKI_LABELS_LIST = ["PERSON", "WORK_OF_ART", "GPE", "ORG", "LOC", "EVENT", "CARDINAL", "NORP", "LANGUAGE"]

# BIO_ANSWER_TYPES = ["GENE", "CHROMOSOME", "GENE_LOCATION", "GENETIC_VARIANT", "INHERITANCE_PATTERN", "PATHWAY", "PROTEIN", "ENZYME", "MOLECULAR_FUNCTION", "CELL_TISSUE_TYPE", "ORGANELLE", "ANATOMY", "DISEASE_CONDITION", "PHENOTYPE_SIGN", "RISK_FACTOR", "PROGNOSIS", "MEDICATION", "MEDICAL_PROCEDURE", "DIAGNOSTIC_TEST", "PATHOGEN", "BIOLOGICAL_TAXON", "CHEMICAL_ENTITY", "POPULATION_GROUP", "TEMPORAL_DATA", "QUANTITY", "PERSON_ORGANIZATION", "YES_NO", "OTHER"]
# WIKI_ANSWER_TYPES = []

def load_prompt_gen_sub(question=None, is_seq=False, dataset=None, type_q=None, context=None):
    if question:
        if is_seq:
            messages = load_subquestion_prompt(question, dataset, type_q)
        else:
            messages = load_search_keyword_prompt(question, dataset, type_q)            
      
    return messages

def load_prompt_gen_ans(question, key_entities, relevant_info, dataset=None, type_q=None):
    if dataset == "medhop":
        prompt = prompt_gen_answer_medhop_step_3b(question, key_entities, relevant_info)
    elif dataset == "wiki":
        prompt = prompt_gen_answer_wiki_step_3b(question, key_entities, relevant_info, type_q=type_q)

    return prompt

def load_search_keyword_prompt(question, dataset=None, type_q=None):
    if dataset == 'medhop':
        labels_str = ", ".join(BIO_LABELS_LIST)
        messages = [
            {
                "role": "system",
                "content": f"""## Role
You are an expert in generating concise and effective Google search queries to help answer complex biomedical questions

## Task
Generate ONLY 1 search query and identify relevant entities

## Instructions
1. Provide a single, keyword-based search query string that best targets the core information needed from the original question
2. Identify a list of target entity types (focus_spacy_based_entity_label) that need to be extracted from the context to answer the question
   - MUST be selected ONLY from this list: [{labels_str}]
3. Identify the specific requested answer type (target_answer_type) using the words from the question
4. Identify a list of specific, concrete, and retrievable entities explicitly mentioned or clearly implied in the question

## Rules
- The query should be a distillation of the original complex question
- Focus on essential nouns and technical terms
- The "key_entity" list should be specific. Exclude generic or abstract concepts (e.g., "medication", "drug", "treatment")

## Constraints
- MUST: Provide ONLY 1 search query
- MUST: Select focus_spacy_based_entity_label ONLY from the following list: [{labels_str}]
- CRITICAL: Map "VIRUS" to "ORGANISM" for focus_spacy_based_entity_label. ALWAYS use "ORGANISM" for viruses
- MUST: Select target_answer_type as a term ONLY from the question

## Output Format (JSON)
[
    {{
        "search_keyword": "Google search query string",
        "focus_spacy_based_entity_label": ["list of labels from the given list"],
        "target_answer_type": "type of answer from question",
        "key_entity": ["list of specific entities from question"]
    }}
]
"""
            },
            {
                "role": "user",
                "content": f"""## Examples (5)

---
**Example 1:**

Question: "Which gene containing a helicase domain is mutated in some patients with Smith-Fineman-Myers syndrome?"

JSON desired output format:
[
    {{ 
        "search_keyword": "Smith-Fineman-Myers syndrome mutated gene helicase domain",
        "focus_spacy_based_entity_label": ["GENE_OR_GENE_PRODUCT"],
        "target_answer_type": "gene",
        "key_entity": [
            "Smith-Fineman-Myers syndrome",
            "helicase domain"
        ]
    }}
]
---
**Example 2:**

Question: "Which lactate dehydrogenase tetramer is most common in the primary organ of the nervous system?"

JSON desired output format:
[
    {{ 
        "search_keyword": "lactate dehydrogenase tetramer most common primary organ nervous system",
        "focus_spacy_based_entity_label": ["CHEMICAL"],
        "target_answer_type": "tetramer",
        "key_entity": [
            "lactate dehydrogenase tetramer",
            "nervous system"
        ]
    }}
]
---
**Example 3:**

Question: "Which physician, along with Edvard Ehlers, is the Ehlers Danlos syndromes named after?"

JSON desired output format:
[
    {{ 
        "search_keyword": "Ehlers Danlos syndromes named after along with Edvard Ehlers",
        "focus_spacy_based_entity_label": ["DISEASE"],
        "target_answer_type": "physician",
        "key_entity": [
            "Ehlers Danlos syndromes",
            "Edvard Ehlers"
        ]            
    }}
]
---
**Example 4:**

Question: "Which benign, autosomal recessive hyperbilirubinemia syndrome is caused by a mutation in the UGT1A1 gene on chromosome 2?"

JSON desired output format:
[
    {{ 
        "search_keyword": "benign autosomal recessive hyperbilirubinemia UGT1A1 gene mutation chromosome 2",
        "focus_spacy_based_entity_label": ["DISEASE", "GENE_OR_GENE_PRODUCT"],
        "target_answer_type": "syndrome",
        "key_entity": [
            "benign autosomal recessive hyperbilirubinemia",
            "UGT1A1 gene",
            "chromosome 2"
        ]
    }}
]
---
**Example 5:**

Question: "Which recognized type of osteochondritis is characterized by uneven vertebral growth and kyphosis that typically develops during adolescence?"

JSON desired output format:
[
    {{ 
        "search_keyword": "osteochondritis type uneven vertebral growth kyphosis adolescence",
        "focus_spacy_based_entity_label": ["DISEASE"],
        "target_answer_type": "type of osteochondritis",
        "key_entity": [
            "osteochondritis",
            "vertebral growth",
            "kyphosis",
            "adolescence"
        ]
    }}
]
---

## Information
Question: "{question}"

## Task
Generate the search query and entity information for the provided question. Output as a JSON array
CRITICAL: You MUST map "VIRUS" to "ORGANISM" in focus_spacy_based_entity_label
"""
            }
        ]
    elif dataset == 'wiki':
        if type_q == "comparison":
            labels_str = ", ".join(WIKI_LABELS_LIST)
            messages = [
                {
                    "role": "system",
                    "content": f"""## Role
You are an expert in generating concise and effective Google search queries

## Task
Generate ONLY 1 search query and identify relevant entities

## Instructions
1. Distill the original complex question into essential keywords and proper names
2. Identify the primary subject (key_entity) extracting it from the question
3. Identify target entity labels and answer type

## Rules
### Relation Keywords
Adapt relation keywords based on the key entity's type to avoid ambiguity:
- For a **Person**: use `country of citizenship`, `date of birth`, `occupation`
- For a **Creative Work** (Film, Album, Song): use `country of origin`, `publication date`, `director`, `producer`
- For a **Location or Organization**: use `country`, `inception`

## Constraints
- MUST: Provide ONLY 1 search query string
- MUST: Omit conversational fillers (e.g., "Who is the")
- MUST: Select focus_spacy_based_entity_label ONLY from: [{labels_str}]
- MUST: Select target_answer_type as a term ONLY from the question

## Output Format (JSON)
[
    {{
        "search_keyword": "generated Google search query string",
        "key_entity": "primary subject extracted from the question",
        "focus_spacy_based_entity_label": ["list of labels from allowed list"],
        "target_answer_type": "type of answer from question"
    }}
]
                    """
                },
                {
                    "role": "user",
                    "content": f"""## Examples (5)

---
**Example 1:**

Question: "Who directed the (1957 Film) 'Just Once A Great Lady'?"

JSON desired output format:
[
    {{
        "search_keyword": "director 'Just Once A Great Lady (1957 Film)'",
        "key_entity": "Just Once A Great Lady (1957 Film)",
        "focus_spacy_based_entity_label": ["PERSON"],
        "target_answer_type": "director"
    }}
]
---
**Example 2:**

Question: "In which country is the film Carnival Of Souls from?"

JSON desired output format:
[
    {{ 
        "search_keyword": "'Carnival Of Souls' country of origin",
        "key_entity": "Carnival Of Souls",
        "focus_spacy_based_entity_label": ["GPE"],
        "target_answer_type": "country"
    }}
]
---
**Example 3:**

Question: "When was the film Engal Aasan released?"

JSON desired output format:
[
    {{
        "search_keyword": "'Engal Aasan' publication date",
        "key_entity": "Engal Aasan",
        "focus_spacy_based_entity_label": ["DATE"],
        "target_answer_type": "date"
    }}
]
---
**Example 4:**

Question: "When was Michelle Tong (Writer) born?"

JSON desired output format:
[
    {{
        "search_keyword": "'Michelle Tong (Writer)' date of birth",
        "key_entity": "Michelle Tong (Writer)",
        "focus_spacy_based_entity_label": ["DATE"],
        "target_answer_type": "date"
    }}
]
---
**Example 5:**

Question: "What is the nationality of the writer William C. White?"

JSON desired output format:
[
    {{
        "search_keyword": "'William C. White' country of citizenship",
        "key_entity": "William C. White",
        "focus_spacy_based_entity_label": ["NORP"],
        "target_answer_type": "nationality"
    }}
]
---

## Information
Question: "{question}"

## Task
Generate the search query and entity information for the provided question. Output as a JSON array
"""
                }
            ]
        else:
            messages = None

    return messages

def load_subquestion_prompt(question, dataset=None, type_q=None):
    if dataset == 'medhop':
        labels_str = ", ".join(BIO_LABELS_LIST)
        messages = [
            {
                "role": "system",
                "content": f"""## Role
You are an expert in biomedical knowledge and complex question decomposition

## Task
Decompose complex multi-hop biomedical questions into a sequence of simpler, interconnected subquestions where each step's answer becomes input for the next step

## Instructions
**Understanding Multi-Hop Reasoning:**
A multi-hop question requires breaking down into subquestions (Q1, Q2, ..., Qn) where:
- Q1 is answered first → produces Answer A1
- A1 is incorporated into Q2 → produces Answer A2
- This chain continues until the final answer is found
- Each subquestion must be answerable independently through search/retrieval

**Example Pattern:**
- Original: "Which chromosome contains the gene encoding the enzyme that inactivates antithrombin?"
- Q1: "Which gene encodes the enzyme that can inactivate antithrombin?" → A1: "SERPINA1 gene"
- Q2: "Which chromosome contains SERPINA1 gene?" → A2: "chromosome 14"

**Decomposition Steps:**
1. Identify information dependencies - determine what intermediate information must be retrieved before the final answer can be found
2. Create a sequential chain where each subquestion builds upon previous answers
3. Ensure each subquestion is self-contained and searchable without needing the answer from previous steps (include full context in the question text)
4. Ensure the last subquestion combines all gathered information to produce the final answer

## Rules
- Question contains nested attributes (e.g., "the gene OF the enzyme THAT does X")
- Question asks about properties of an unknown entity (must first identify the entity)
- Question requires connecting multiple pieces of information sequentially
- Subquestions MUST be searchable by standard query engines (Google, Wikipedia, PubMed)
- Subquestions MUST include all necessary context explicitly (don't rely on implicit information from previous steps)
- Avoid pronouns or vague references (use "that gene", "that enzyme", "that syndrome" instead of "it")
- Each subquestion should yield a single, specific entity when possible (not a list)
- Avoid generating lists in intermediate steps when possible
- If a step must produce multiple entities, the next step should specify how to filter/select (e.g., "Which of those genes is located on chromosome 7?")
- Use the MINIMUM number of steps necessary to answer the question
- MAXIMUM 3 hops (subquestions) allowed - do not exceed this limit
- First step search_keyword: Include all relevant context from the original question
- Dependent step search_keyword: Generate ONLY keywords for the CURRENT step without repeating previous hop context
- key_entity: List specific, retrievable biomedical entities only
- EXCLUDE generic concepts like "medication", "drug", "treatment", "disease" unless they are specific named entities

## Constraints
- MUST: Use ONLY the following keys for each step: "subquestion", "search_keyword", "focus_spacy_based_entity_label", "target_answer_type", "key_entity"
- MUST: Select focus_spacy_based_entity_label ONLY from the following list: [{labels_str}]
- MUST: For target_answer_type, extract a notable short phrase from the question (e.g., "gene", "chromosome", "syndrome"). Exception: Use "yes_no" for yes/no questions
- CRITICAL: Map "VIRUS" to "ORGANISM" for focus_spacy_based_entity_label. ALWAYS use "ORGANISM" for viruses

## Output Format (JSON)
[
    {{
        "subquestion": "subquestion",
        "search_keyword": "distilled, concise keyword-based query",
        "focus_spacy_based_entity_label": ["list of labels from allowed list"],
        "target_answer_type": "type of answer from question",
        "key_entity": ["specific retrievable entities"]
    }}
]

"""
            },
            {
                "role": "user",
                "content": f"""## Examples (5)

---
**Example 1:**
"Which human chromosome contains the gene that encodes the enzyme that can inactivate antithrombin?"

JSON desired output format:
[
    {{
        "subquestion": "Which gene encode enzyme can inactivate antithrombin?",
        "search_keyword": "gene encode enzyme inactivate antithrombin",
        "focus_spacy_based_entity_label": ["GENE_OR_GENE_PRODUCT"],
        "target_answer_type": "gene",
        "key_entity": ["antithrombin"]
    }},
    {{
        "subquestion": "Which human chromosome contains that gene?",
        "search_keyword": "human chromosome location gene",
        "focus_spacy_based_entity_label": ["CELLULAR_COMPONENT"],
        "target_answer_type": "chromosome",
        "key_entity": []
    }}
]

---
**Example 2:**
"Which lactate dehydrogenase tetramer is most common in the primary organ of the nervous system?"

JSON desired output format:
[
    {{
        "subquestion": "What is the primary organ of the human nervous system?",
        "search_keyword": "primary organ human nervous system",
        "focus_spacy_based_entity_label": ["ORGAN"],
        "target_answer_type": "organ",
        "key_entity": ["nervous system"]
    }},
    {{
        "subquestion": "Which specific lactate dehydrogenase tetramer is most commonly found in that organ?",
        "search_keyword": "lactate dehydrogenase tetramer most common organ",
        "focus_spacy_based_entity_label": ["CHEMICAL"],
        "target_answer_type": "tetramer",
        "key_entity": []
    }}
]

---
**Example 3:**
"Which gene mutation is responsible for the syndrome characterized by flattened and deformed vertebrae, developmental delay, and lens coloboma?"

JSON desired output format:
[
    {{
        "subquestion": "Which syndrome is characterized by flattened and deformed vertebrae, developmental delay, and lens coloboma?",
        "search_keyword": "syndrome flattened deformed vertebrae developmental delay lens coloboma",
        "focus_spacy_based_entity_label": ["DISEASE"],
        "target_answer_type": "syndrome",
        "key_entity": [
            "flattened vertebrae",
            "deformed vertebrae",
            "developmental delay",
            "lens coloboma"
        ]
    }},
    {{
        "subquestion": "Which gene mutation is responsible for that syndrome?",
        "search_keyword": "gene mutation syndrome",
        "focus_spacy_based_entity_label": ["GENE_OR_GENE_PRODUCT"],
        "target_answer_type": "gene mutation",
        "key_entity": []
    }}
]

---
**Example 4:**
"Which London suburb was home to the pioneering institutions that influenced John Langdon Down's work at Earlswood Asylum?"

JSON desired output format:
[
    {{
        "subquestion": "Which pioneering institutions influenced John Langdon Down's work at Earlswood Asylum?",
        "search_keyword": "pioneering institutions John Langdon Down Earlswood Asylum",
        "focus_spacy_based_entity_label": ["ORG"],
        "target_answer_type": "institution",
        "key_entity": [
            "John Langdon Down",
            "Earlswood Asylum"
        ]
    }},
    {{
        "subquestion": "In which London suburb was that pioneering institution located?",
        "search_keyword": "London suburb location institution",
        "focus_spacy_based_entity_label": ["GPE"],
        "target_answer_type": "suburb",
        "key_entity": []
    }}
]

---
**Example 5:**
"On which chromosome is the gene responsible for Carpenter syndrome, which is associated with brachydactyly, located?"

JSON desired output format:
[
    {{
        "subquestion": "Which gene is responsible for Carpenter syndrome?",
        "search_keyword": "gene responsible Carpenter syndrome",
        "focus_spacy_based_entity_label": ["GENE_OR_GENE_PRODUCT"],
        "target_answer_type": "gene",
        "key_entity": [
            "Carpenter syndrome"
        ]
    }},
    {{
        "subquestion": "On which chromosome is that gene located?",
        "search_keyword": "chromosome location gene",
        "focus_spacy_based_entity_label": ["CELLULAR_COMPONENT"],
        "target_answer_type": "chromosome",
        "key_entity": []
    }}
]

## Information
Original Complex Question: '{question}'

## Output Format (JSON)
[
    {{
        "subquestion": "subquestion",
        "search_keyword": "distilled, concise keyword-based query",
        "focus_spacy_based_entity_label": ["list of labels from allowed list"],
        "target_answer_type": "type of answer from question",
        "key_entity": ["specific retrievable entities"]
    }}
]

CRITICAL: You MUST map "VIRUS" to "ORGANISM" in focus_spacy_based_entity_label
"""
            }
            ]
    elif dataset == "wiki":
        labels_str = ", ".join(WIKI_LABELS_LIST)
        if type_q == "bridge_comparison":
            messages = [
                {
                    "role": "system",
                    "content": f"""## Role
You are a highly specialized AI assistant and expert at decomposing complex questions

## Task
Decompose complex questions into a sequence of simpler, interconnected subquestions to create a logical reasoning plan

## Instructions
1. **Faithful Reasoning via Decomposition**: Externalize the reasoning process by breaking the question into a series of subquestions
2. **Sequential Planning**: Generate a sequence of subquestions where each step depends on the previous ones
3. **Template Usage**: Use placeholders when a subquestion depends on a previous result (e.g., "that director", "that person")

## Rules
### Dynamic Planning
- Subquestions can be self-contained or templates
- Placeholders (e.g., "the entity found in the previous step") indicate where programmatic insertion will occur
- Search keywords can also be templates (e.g., "director date of birth") intended for combination with prior results

### Efficiency
- Decompose into the MINIMUM number of steps necessary

## Constraints
- MUST: Select focus_spacy_based_entity_label ONLY from: [{labels_str}]
- MUST: Select target_answer_type as a term ONLY from the question

## Output Format (JSON)
[
    {{
        "subquestion": "subquestion or template",
        "search_keyword": "keyword-based query or template",
        "key_entity": "specific entity name or empty for dependent steps",
        "focus_spacy_based_entity_label": ["list of labels from allowed list"],
        "target_answer_type": "type of answer from question"
    }}
]

Original Complex Question: '{question}'
"""
                },
                {
                    "role": "user",
                    "content": f"""## Examples (6)

---
**Example 1:**
"What is the country of the director of film Move (1970 Film)"

JSON desired output format:
[
    {{
        "subquestion": "What is the director of film Move (1970 Film)?",
        "search_keyword": "director 'Move (1970 Film)'",
        "key_entity": "Move (1970 Film)",
        "focus_spacy_based_entity_label": ["PERSON"],
        "target_answer_type": "director"
    }},
    {{
        "subquestion": "What country of citizenship is that film director from?",
        "search_keyword": "country of citizenship",
        "key_entity": "",
        "focus_spacy_based_entity_label": ["GPE"],
        "target_answer_type": "country"
    }}
]
---
**Example 2:**
"When was the director of film Blindfold (1966 Film) born?"

JSON desired output format:
[
    {{
        "subquestion": "Who directed the (1966 Film) 'Blindfold'?",
        "search_keyword": "director 'Blindfold (1966 Film)'",
        "key_entity": "Blindfold (1966 Film)",
        "focus_spacy_based_entity_label": ["PERSON"],
        "target_answer_type": "director"
    }},
    {{
        "subquestion": "When was that person born?",
        "search_keyword": "date of birth",
        "key_entity": "",
        "focus_spacy_based_entity_label": ["DATE"],
        "target_answer_type": "date"
    }}
]
---
**Example 3:**
"What is the date of death of Helen F. Holt's husband?"

JSON desired output format:
[
    {{
        "subquestion": "Who was the husband of Helen F. Holt?",
        "search_keyword": "'Helen F. Holt' husband",
        "key_entity": "Helen F. Holt",
        "focus_spacy_based_entity_label": ["PERSON"],
        "target_answer_type": "husband"
    }},
    {{
        "subquestion": "What is the date of death of that person?",
        "search_keyword": "date of death",
        "key_entity": "",
        "focus_spacy_based_entity_label": ["DATE"],
        "target_answer_type": "date"
    }}
]
---
**Example 4:**
"Where did the director of film The Right Of The Strongest die?"

JSON desired output format:
[
    {{
        "subquestion": "Who was the director of the film 'The Right Of The Strongest'?",
        "search_keyword": "director film 'The Right Of The Strongest'",
        "key_entity": "The Right Of The Strongest",
        "focus_spacy_based_entity_label": ["PERSON"],
        "target_answer_type": "director"
        
    }},
    {{
        "subquestion": "Where did that person die?",
        "search_keyword": "place of death",
        "key_entity": "",
        "focus_spacy_based_entity_label": ["GPE"],
        "target_answer_type": "place"
    }}
]
---
**Example 5:**
"Who is the spouse of the director of film Just Once A Great Lady (1957 Film)?"

JSON desired output format:
[
    {{
        "subquestion": "Who directed the 'Just Once A Great Lady (1957 Film)'?",
        "search_keyword": "director 'Just Once A Great Lady (1957 Film)'",
        "key_entity": "Just Once A Great Lady (1957 Film)",
        "focus_spacy_based_entity_label": ["PERSON"],
        "target_answer_type": "director"
    }},
    {{
        "subquestion": "Who was the spouse of that person?",
        "search_keyword": "spouse of",
        "key_entity": "",
        "focus_spacy_based_entity_label": ["PERSON"],
        "target_answer_type": "spouse"
    }}
]
---
**Example 6:**
"When was the producer of "My Friends Told Me About You" born?"
JSON desired output format:
[
    {{
        "subquestion": "Who produced the 'My Friends Told Me About You'?",
        "search_keyword": "producer 'My Friends Told Me About You'",
        "key_entity": "My Friends Told Me About You",
        "focus_spacy_based_entity_label": ["PERSON"],
        "target_answer_type": "producer"
    }},
    {{
        "subquestion": "When was that person born?",
        "search_keyword": "date of birth",
        "key_entity": "",
        "focus_spacy_based_entity_label": ["DATE"],
        "target_answer_type": "date"
    }}
]

## Information
Original Complex Question: '{question}'

## Task
Decompose the question into subquestions following the examples and rules. Output in JSON format
"""
                }
            ]
        elif type_q == "compositional":
            messages = [
                {
                    "role": "system",
                    "content": f"""## Role
You are a highly specialized AI assistant and expert in analyzing complex questions with hidden sequential logic, particularly in social and organizational structures (e.g., family trees, corporate hierarchies)

## Task
Decompose complex questions into a sequence of simpler, interconnected subquestions to create a logical reasoning plan.

## Instructions
1. **Faithful Reasoning via Decomposition**: Externalize the reasoning process by breaking the question into a series of subquestions
2. **Sequential Planning**: Generate a sequence of subquestions where each step depends on the previous ones
3. **Template Usage**: Use placeholders when a subquestion depends on a previous result (e.g., "that director", "that person")

## Rules
### Dynamic Planning
- Subquestions must be templates that depend on prior answers
- Use clear placeholders (e.g., "the entity found in the previous step") for programmatic insertion
- Search keywords are also templates designed to be combined with prior results

### Efficiency
- Decompose into the MINIMUM number of steps necessary

## Constraints
- MUST: Select focus_spacy_based_entity_label ONLY from: [{labels_str}]
- MUST: Select target_answer_type as a term ONLY from the question

## Output Format (JSON)
[
    {{
        "subquestion": "subquestion or template",
        "search_keyword": "keyword-based query or template",
        "key_entity": "specific entity name or empty for dependent steps",
        "focus_spacy_based_entity_label": ["list of labels from allowed list"],
        "target_answer_type": "type of answer from question"
    }}
]

Original Complex Question: '{question}'
"""
                    },
                    {
                        "role": "user",
                        "content": f"""## Examples (4)

---
**Example 1:**
"Which award did the director of the movie Parasite receive at the Oscars?"
JSON desired output format:
[
{{
    "subquestion": "Who is the director of the movie Parasite?",
    "search_keyword": “director of the movie 'Parasite'",
    "key_entity": “Parasite",
    "focus_spacy_based_entity_label": ["PERSON"],
    "target_answer_type": "director"
}},
{{
    "subquestion": "Which award did that director receive at the Oscars?",
    "search_keyword": "award receive at Oscars",
    "key_entity": "",
    "focus_spacy_based_entity_label": ["WORK_OF_ART"],
    "target_answer_type": "award"
}}
]

---
**Example 2:**
When is Ada Lovelace (1815–1852)'s father birthday?
JSON desired output format:
[
{{
    "subquestion": "Who was the father of Ada Lovelace (1815–1852)?",
    "search_keyword": "Ada Lovelace (1815–1852) father",
    "key_entity": "Ada Lovelace (1815–1852)",
    "focus_spacy_based_entity_label": ["PERSON"],
    "target_answer_type": "father"
}},
{{
    "subquestion": "When is that person date of birth?",
    "search_keyword": "date of birth",
    "key_entity": "",
    "focus_spacy_based_entity_label": ["DATE"],
    "target_answer_type": "date"
}}
]
---
**Example 3:**
"Where was the director of film Doctor Strange (2016 film) born"
JSON desired output format:
[
{{
    "subquestion":"Who was the director of film Doctor Strange (2016 film)",
    "search_keyword": "director of film 'Doctor Strange (2016 film)'",
    "key_entity": "Doctor Strange (2016 film)",
    "focus_spacy_based_entity_label": ["PERSON"],
    "target_answer_type": "director"
}},
{{
    "subquestion": "Where was director born?",
    "search_keyword": "place of birth",
    "key_entity": "",
    "focus_spacy_based_entity_label": ["GPE"],
    "target_answer_type": "place"
}}
]
---
**Example 4:**
"Which country the director of film Web Of Suspicion is from?"
JSON desired output format:
[
{{
    "subquestion":"Who was the director of film Web Of Suspicion?",
    "search_keyword": "director of film 'Web Of Suspicion'",
    "key_entity": "Web Of Suspicion",
    "focus_spacy_based_entity_label": ["PERSON"],
    "target_answer_type": "director"
}},
{{
    "subquestion": "Which country of citizenship is that director from?",
    "search_keyword": "country of citizenship",
    "key_entity": "",
    "focus_spacy_based_entity_label": ["GPE"],
    "target_answer_type": "country"
}}
]

## Information
Original Complex Question: '{question}'

## Task
Decompose the question into subquestions following the examples and rules. Output in JSON format
"""
                }
            ]
        elif type_q == "inference":
            messages = [
                {
                    "role": "system",
                    "content": f"""## Role
You are an expert in identifying key entities and relations in questions.

## Task
Identify the key entity, the key relation, target entity labels, and the answer type for the given question

## Instructions
1. Analyze the question to find the main subject (key_entity)
2. Identify the core relation being asked about (key_relation)
3. Determine the expected answer type and entity labels

## Constraints
- MUST: Select focus_spacy_based_entity_label ONLY from: [{labels_str}]
- MUST: Select target_answer_type as a term ONLY from the question

## Output Format (JSON)
[
    {{
        "search_keyword": "Distilled search query",
        "key_entity": "primary subject",
        "key_relation": "core relation",
        "focus_spacy_based_entity_label": ["list of labels from allowed list"],
        "target_answer_type": "type of answer from question"
    }}
]
"""
                },
                {
                    "role": "user",
                    "content": f"""## Examples (4)

---
**Example 1:**

Question: "Who is Giselda Volodi's sibling-in-law?"

JSON desired output format:
[
    {{
        "search_keyword": "'Giselda Volodi' sibling-in-law",
        "key_entity": "Giselda Volodi",
        "key_relation": "sibling-in-law",
        "focus_spacy_based_entity_label": ["PERSON"],
        "target_answer_type": "sibling-in-law"
    }}
]

---
**Example 2:**

Question: "Who is Humphrey Of Montfort's maternal grandmother?"

JSON desired output format:
[
    {{
        "search_keyword": "'Humphrey Of Montfort' maternal grandmother",
        "key_entity": "Humphrey Of Montfort",
        "key_relation": "maternal grandmother",
        "focus_spacy_based_entity_label": ["PERSON"],
        "target_answer_type": "grandmother"
    }}
]

---
**Example 3:**

Question: "Who is the aunt of Margaretta Taylor?"

JSON desired output format:
[
    {{
        "search_keyword": "'Margaretta Taylor' aunt",
        "key_entity": "Margaretta Taylor",
        "key_relation": "aunt",
        "focus_spacy_based_entity_label": ["PERSON"],
        "target_answer_type": "aunt"
    }}
]

---
**Example 4:**

Question: "What is the field of work of Fred Diamond?"

JSON desired output format:
[
    {{
        "search_keyword": "'Fred Diamond' field of work",
        "key_entity": "Fred Diamond",
        "key_relation": "field of work",
        "focus_spacy_based_entity_label": ["ORG"],
        "target_answer_type": "field of work"
    }}
]

---

## Information
Question: "{question}"

## Task
Identify the key entity, key relation, target entity labels, and target answer type. Output in JSON format
"""
                }
            ]

    return messages


def prompt_gen_answer_medhop_step_3b(question, key_entities, context):
    prompt = f"""## Role
You are an expert in biomedical knowledge.

## Task
Answer the following biological question based on the provided information.

## Instructions
1. Analyze the provided context carefully
2. Generate two parts in JSON format: a detailed explanation and a short answer

## Rules
### Short Answer Format
- **Count questions** ("how many"): Return ONLY a number (e.g., "10", "ten")
- **Time/Duration questions**: Return a concise time value (e.g., "3 days")
- **Yes/No questions**: Return exactly "Yes" or "No"
- **Entity questions**: Return the full noun phrase naming the entity (e.g., "chromosome X")
- **Organism questions**: Return organism type (e.g., "bacterium")
- **Insufficient Info**: If the information is missing, return exactly "Insufficient information"

## Constraints
- MUST: Your short_answer MUST be MAX 3 words
- MUST NOT: Add any extra commentary outside the JSON
- MUST NOT: Use abbreviations for entities if the full name is available

## Information
Question: {question}
Key entities: {key_entities}
Context: {context}

## Output Format (JSON)
{{
    "long_answer": "detailed explanation based on context",
    "short_answer": "1-3 word answer following the rules"
}}
"""
    return prompt


def prompt_gen_answer_wiki_step_3b(question, key_entities, context, type_q=None):
    if type_q in ["bridge_comparison", "comparison"]:
        prompt = f"""## Role
You are an expert analyst.

## Task
Answer the following question based on the provided information.

## Instructions
1. Analyze the question and identify the core relation being asked
2. Extract the direct value from the provided context
3. Format the response as a valid JSON object

## Rules
### Answer Logic
- **Yes/No Questions**: Return "Yes" or "No"
- **Identity Questions**: Identify the person or entity matching the role/description and return their name as found in context
- **Insufficient Info**: If the information is missing, return exactly "Insufficient information"

## Constraints
- MUST: Provide ONLY the JSON object
- MUST NOT: Add commentary before or after the JSON
- MUST NOT: Add extra parenthetical information in the short_answer

## Information
Question: {question}
Key entities: {key_entities}
Context: {context}

## Output Format (JSON)
{{
    "long_answer": "Detailed reasoning explaining how the answer was derived",
    "short_answer": "The most direct answer from the context"
}}
"""
    elif type_q == "inference":
        prompt = f"""## Role
You are an expert in historical and genealogical analysis

## Task
Answer the question by identifying the correct person and precisely formatting their name and titles

## Instructions
1. Analyze the context to find the requested person
2. Construct their full name and titles following the specific rules for royal/noble figures
3. Split the resulting answer into prefix, name, and suffix components

## Rules
### Rule 1: Royal & Noble Figures
- **Start with**: Full name and regnal number
- **Include**: Hereditary titles, noble/royal prefixes (King, Queen, Duke, Sir, etc.)
- **Exclude**: Territorial designations based ONLY on land ownership
- **Parentheses**: Do not include parentheses, but use their content for extraction

### Rule 2: Non-Royal Figures
- Provide the single most direct answer found in the context

### Rule 3: Insufficient Information
- If the answer is missing, use "Insufficient information" for both the short_answer and the name field. Set prefix and suffix to "NAN"

## Constraints
- MUST: Your response must be ONLY the JSON object
- MUST NOT: Include commentary outside the JSON
- MUST: Follow the name splitting rules precisely

## Information
Question: {question}
Key entities: {key_entities}
Context: {context}

## Output Format (JSON)
{{
    "long_answer": "detailed explanation showing the derivation",
    "short_answer": "full answer string as per rules",
    "prefix": "titles before name or NAN",
    "name": "core personal name and regnal number",
    "suffix": "titles or post-nominals after name or NAN"
}}
"""
    elif type_q == "compositional":        
        prompt = f"""## Role
You are an expert intelligence analyst

## Task
Synthesize information across multiple documents to provide the single most probable answer to the question

## Instructions
1. **Context Synthesis**: Read ALL provided documents and connect information across them
2. **Entity Resolution**: Treat synonyms or alternate names as the same entity
3. **Evidence Priority**: Decide if the answer is Explicit (stated) or Inferred (logical deduction). Use explicit evidence first
4. **Mandatory Inference**: If no explicit answer exists, you are REQUIRED to infer from strong clues (residence, professional role, nationality fallback)

## Rules
### Location Questions
- Return the MOST SPECIFIC location available (city/town > state/province > country)
- If birthplace/death place is missing, use nationality (country) as a fallback and mark as inferred

### Category Questions
- Choose the broadest applicable category (e.g., "cancer" not "bowel cancer")

### Nationality Formatting
- "X-born Y" → answer "X"
- "X-Y" nationality → answer "X"
- "X Y" nationality → answer "Y"

## Constraints
- MUST: Select ONLY the single most probable answer
- MUST NOT: Generate names of locations (cities, regions, countries) NOT present in the context
- MUST NOT: Add any additional commentary outside the JSON

## Information
Question: {question}
Key entities: {key_entities}
Context: {context}

## Output Format (JSON)
{{
    "long_answer": "Complete reasoning clearly stating if explicit or inferred, citing specific clues/documents used",
    "short_answer": "The single most probable answer following the formatting rules"
}}
"""
          
    return prompt


def prompt_gen_combine_parallel_answers(question, context):
    prompt = f"""## Role
You are an expert in logical synthesis and comparative analysis.

## Task
Analyze the provided information to determine the correct answer and reconcile any potential differences across sources.

## Instructions
1. Determine the correct answer based on the provided context
2. Generate a valid JSON object with detailed reasoning (long_answer) and a direct answer (short_answer)

## Rules
### Choice Questions
- If the question asks to choose an option based on a criterion (e.g., "younger", "first", "later"):
- The short_answer MUST be the name of the chosen option extracted EXACTLY from the question

### Confirmation Questions
- If the question asks to confirm a fact (e.g., "same country", "same nationality"):
- The short_answer MUST be exactly "yes" or "no" (lowercase)

## Constraints
- MUST: Your entire response must be ONLY the JSON object
- MUST NOT: Include commentary outside the JSON

## Information
Question: {question}
Context: {context}

## Output Format (JSON)
{{
    "long_answer": "Detailed explanation synthesized from context",
    "short_answer": "The final answer following the rules"
}}
"""
    return prompt

import json
def prompt_gen_relation(list_subquestion, type_q):
    if type_q == "inference":
        allowed_relations = {
            'father', 'mother', 'sibling', 'spouse', 'child',
            'doctoral student', 'educated at', 'field of work',
            'doctoral advisor', 'employer'
        }
    elif type_q == "compositional":
        allowed_relations = {
            'father', 'mother', 'sibling', 'spouse',
            'director', 'performer', 'composer',
            'educated at', 'founded by',
            'date of birth', 'place of birth',
            'date of death', 'place of death', 'cause of death',
            'place of burial',
            'country', 'country of origin', 'country of citizenship',
            'publication date'
        }
    elif type_q == "comparison":
        allowed_relations = set()
        data = {
            'Human': ['date of birth', 'date of death', 'date of birth and date of death (year old)', 'occupation', 'country of citizenship', 'place of birth'],
            'Film': ['publication date', 'director', 'producer', 'country of origin'],
            'Album': ['publication date', 'producer'],
            'Musical group': ['inception', 'country of origin'],
            'Song': ['publication date'],
            'General entities': ['inception', 'country'],
            'Topography': ['country']
        }
        for relations_list in data.values():
            allowed_relations.update(relations_list)

    elif type_q == "bridge_comparison":
        allowed_relations = {
            "director date of birth", "director date of death", "director country of citizenship",
            "producer date of birth", "producer date of death", "producer country of citizenship"
        }
    else:
        raise ValueError("Invalid question type. Please choose one of the following: inference, compositional, comparison, bridge_comparison.")

    system_prompt = f"""## Role
You are a highly specialized relation extraction tool.

## Task
Identify the single relevant relation for each subquestion provided in the list.

## Instructions
1. Review the provided list of subquestions and the question type
2. Map each subquestion to a single relation from the allowed set
3. Maintain the order and count of the input list in your output

## Rules
- **Allowed Relations**: Select ONLY from this set: {list(allowed_relations)}
- **One-to-One Mapping**: The number of relations in the output MUST exactly match the number of subquestions

## Constraints
- MUST: Your ONLY output must be a JSON list of strings
- MUST NOT: Include any additional text, explanations, or commentary

## Examples
---
**Example 1:**
Input: {{
    "list_subquestion": ["Who directed Parasite?", "When was Parasite published?"],
    "type_q": "compositional"
}}
Output: ["director", "publication date"]
---
**Example 2:**
Input: {{
    "list_subquestion": ["Who is the father of William Gore?", "Who is the father of that person?"],
    "type_q": "inference"
}}
Output: ["father", "father"]
---

## Output Format (JSON)
["relation_1", "relation_2", ...]
"""

    user_prompt = {
        "list_subquestion": list_subquestion,
        "type_q": type_q
    }

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False, indent=2)}
    ]

    return messages

def prompt_split_parallel_question_medhop(question):
    system_prompt = """## Role
You are an expert in biomedical question decomposition

## Task
Split a complex comparative question into independent subquestions based on the entities being compared

## Constraints
- MUST: Output ONLY a valid array of strings in JSON format
- MUST NOT: Include any explanation, introductory text, or markdown formatting
- MUST NOT: Change the core entities or the relation being compared
- MUST: Ensure subquestions are complete and context-rich enough to be answered independently"""

    user_prompt = f"""## Task
Decompose the provided complex comparative question into retrieval-focused subquestions

## Instructions
1. Identify all primary entities being compared in the question
2. Identify the specific attribute or property being compared
3. Create independent questions that each ask for that attribute for each entity identified
4. Format the final result as a simple array of strings

## Information
Question: "{question}"

## Examples
---
**Example 1:**
Input: "Can patients exhibiting symptoms of either aniridia, genitourinary anomalies, or intellectual disability also be at an increased risk for developing nephroblastoma?"
Output:
[
    "Are patients with aniridia at an increased risk for developing nephroblastoma?",
    "Are patients with genitourinary anomalies at an increased risk for developing nephroblastoma?",
    "Are patients with intellectual disability at an increased risk for developing nephroblastoma?"
]
---
**Example 2:**
Input: "What genetic variations are linked to cardiovascular abnormalities, specifically are they found in MYH7 or TTN?"
Output:
[
    "Which genetic variations in MYH7 are linked to cardiovascular abnormalities?",
    "Which genetic variations in TTN are linked to cardiovascular abnormalities?"
]
---

## Output Format (JSON)
["sub_question_1", "sub_question_2", "..."]
"""
        
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]
         
    return messages

def prompt_split_parallel_question_wiki(question):
    system_prompt = """## Role
You are an expert in question decomposition

## Task
Split a complex comparative question into exactly two simple subquestions

## Constraints
- MUST: Output ONLY a valid array of strings in JSON format
- MUST NOT: Include any explanation, introductory text, or markdown formatting
- MUST NOT: Change the core entities or the relation being compared
- MUST: Ensure subquestions are complete and context-rich enough to be answered independently"""

    user_prompt = f"""## Task
Decompose the provided complex comparative question into two retrieval-focused subquestions

## Instructions
1. Identify the two primary entities being compared in the question
2. Identify the specific attribute or property (e.g., nationality, date of birth, location) being compared
3. Create two independent questions that each ask for that attribute for one of the entities
4. Format the final result as a simple array of strings

## Information
Question: "{question}"

## Examples
---
**Example 1:**
Input: "Do both directors of films The Bird of Happiness (film) and Bunco Squad share the same nationality?"
Output:
[
    "What is the nationality of the director of 'The Bird of Happiness (film)'?",
    "What is the nationality of the director of 'Bunco Squad'?"
]
---
**Example 2:**
Input: "Which film has the director who is older than the other, Tiny Times 3 or Being Rose?"
Output:
[
    "What is the date of birth of the director of 'Tiny Times 3'?",
    "What is the date of birth of the director of 'Being Rose'?"
]
---

## Output Format (JSON)
["sub_question_1", "sub_question_2"]
"""
        
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]
         
    return messages


def get_answer_generation_system_prompt(dataset: Optional[str] = None) -> str:
   """
   Get system message for answer generation with injected rules.
  
   Args:
       dataset: Dataset context (medhop, bioasq, cwq, wiki)
      
   Returns:
       System message string
   """
   if dataset in ["medhop", "bioasq"]:
       domain_context = "You are an expert in medical and scientific knowledge."
   else:
       domain_context = "You are an expert in general knowledge."
  
   return f"""## Role
{domain_context}

## Task
Answer the question based on the provided context following precise formatting rules

## Instructions
1. **Analyze Context**: Read all documents carefully and connect facts
2. **Determine Evidence**: Identify explicit facts first, then make logical inferences
3. **Synthesis**: Treat synonyms/alternate names as the same entity
4. **Formatting**: Apply the EXACT formatting rules provided for your specific question type

## Rules
### Context Synthesis
- Prioritize explicit evidence over inference
- If no explicit answer exists, infer from strong clues:
  - Long-term residence/workplace (strongest clue)
  - Professional role implied by context
  - Place of birth/nationality as fallback for location questions
- Mark inferred answers clearly in the long_answer

### Constraints
- MUST: Select ONLY the SINGLE most probable answer
- MUST: short_answer must be MAX 3 words
- MUST NOT: Generate information (especially locations, names) not present in context
- MUST NOT: Include commentary outside the JSON
"""

def get_answer_generation_context_synthesis_rules() -> str:
    """
    Get context synthesis rules for answer generation.
    
    Returns:
        Rules section string for prompt injection
    """
    return """
### CONTEXT SYNTHESIS RULES ###
- Read ALL provided context carefully
- Connect information across documents
- Treat synonyms/alternate names as the same entity
- Prioritize explicit evidence over inference
- Do NOT generate information (especially locations, names) not present in context
- If no explicit answer exists, infer from strong clues:
  * Long-term residence/workplace (strongest clue)
  * Professional role implied by context
  * Place of birth/nationality as fallback for location questions
- Mark inferred answers clearly in long_answer
"""

def get_answer_generation_type_specific_rules(type_q: Optional[str] = None, dataset: Optional[str] = None) -> str:
   """
   Get type-specific rules for answer generation based on question type and dataset.
  
   For MedHopQA (medhop): Returns biomedical-specific rules.
   For 2WikiMultiHopQA (wiki): Returns rules based on type_q (inference, compositional,
   comparison, bridge_comparison).
  
   Args:
       type_q: Question type for 2WikiMultiHopQA (inference, compositional, comparison,
               bridge_comparison). Not used for MedHopQA.
       dataset: Dataset context (medhop, wiki)
      
   Returns:
       Type-specific rules string for prompt injection
   """
   # =========================================================================
   # MEDHOP (Biomedical Domain)
   # =========================================================================
   if dataset == "medhop":
       return """
## Biomedical Domain Rules

### Answer Format by Question Pattern

1. COUNT Questions ("how many", "number of"):
   - Return ONLY a number or numeric phrase
   - Examples: '10', 'ten', 'twenty-five'

2. TIME/DURATION Questions ("how long", "duration"):
   - Return concise time value with units
   - Examples: '36 hours', '3 days', '2 weeks'

3. YES/NO Questions ("is", "are", "does", "can"):
   - Return exactly 'Yes' or 'No'
   - No additional explanation in short_answer

4. ENTITY Questions ("what", "which"):
   - Return the FULL biomedical noun phrase
   - Include complete terminology, not abbreviations alone
   - Examples: 'chromosome X', 'interleukin-6', 'dopamine receptor D2'

5. ORGANISM Questions ("what organism", "what type"):
   - Return the organism type/classification
   - Examples: 'bacterium', 'fungus', 'virus', 'mammal'

6. RANGE/INTERVAL Questions ("range", "between"):
   - Return numeric range with units
   - Examples: '300-330 msec', '10-20%', '5-10 days'

7. DEFAULT (other questions):
   - Provide the most direct, specific answer from context
   - Use complete biomedical terminology

### Constraints
- MUST: Select ONLY the SINGLE most probable answer
- MUST: MAX 3 words for short_answer
"""

   if dataset == "wiki":
       if type_q == "inference":
           return """
## Inference Type Rules

This question involves family relations or royal/noble figures

### For Royal/Noble Figures
- Return the full name with royal prefixes (King, Queen, Prince, Princess, Duke, Duchess, Sir, Dame, Lord, Lady)
- Include numbered hereditary titles if present
- MUST NOT: Include territorial designations based only on land ownership, or parentheses
- Return as a SINGLE string (MAX 3 words)

### Examples
- "King William IV"
- "Princess Luisa"
- "Sir Thomas Monck"

### For Non-Royal Figures
- Return the full name as found in context (MAX 3 words)
"""

       elif type_q == "compositional":
           return """
## Compositional Type Rules

This question requires connecting facts across multiple documents.

### Core Requirements
- MUST: Synthesize information from different sources
- Treat synonyms/alternate names as the same entity
  - Example: "John Doe" = "Jean Doe" = same person

### Location Answers
- Return MOST SPECIFIC location available (city > state > country)
- MUST NOT: Generate locations not present in context
- If only nationality is available, use as country fallback and mark as INFERRED

### Nationality Answers
- "X-born Y" (e.g., "Spanish-born Brazilian") → answer is "X" (Spanish)
- "X-Y" nationality (e.g., "Mexican-American") → answer is "X" (Mexican)
- "X Y" nationality (e.g., "German Italian") → answer is "Y" (Italian)

### General Categories
- Use the broadest applicable term
- Example: "cancer" instead of "bowel cancer"
"""

       elif type_q in ["comparison", "bridge_comparison"]:
           return """
## Comparison Type Rules

This question compares two or more entities.

### Choice Questions (Which/Who is older/younger/later/earlier/first/last)
- Return the NAME of the chosen option extracted EXACTLY from the question
- MUST NOT: Return computed values like dates
- Example: "Which film has director born later, A Flame In My Heart or Butcher, Baker?"
  → Answer is "A Flame In My Heart" or "Butcher, Baker", NOT a date

### Confirmation Questions (Are/Do both X and Y the same...)
- Return 'yes' or 'no' (lowercase only)
- MUST NOT: Include additional explanation in short_answer
- Example: "Are both directors from the same country?" → "yes" or "no"
"""
   # No specific rules for unknown dataset/type
   return ""


def get_answer_generation_output_format(type_q: Optional[str] = None, is_royal_noble: bool = False) -> str:
    if type_q == "inference" or is_royal_noble:
        return """{
    "long_answer": "Step-by-step explanation with evidence citations",
    "short_answer": "Answer following formatting rules, SINGLE most probable answer (NO lists/separators)",
    "prefix": "Royal/noble prefix or NAN",
    "name": "Core personal name",
    "suffix": "Titles after name or NAN"
}"""
    else:
        return """{
    "long_answer": "Step-by-step explanation with evidence citations",
    "short_answer": "Answer following formatting rules exactly, SINGLE most probable answer (NO lists/separators)"
}"""


def get_answer_generation_user_prompt(
    question: str,
    context_str: str,
    rules_text: str,
    output_format: str,
    q_key_entity: Optional[str] = None,
    evidence_key_entity: Optional[str] = None
) -> str:
    
    key_entity_text = f"\nKey Entity: {evidence_key_entity}" if evidence_key_entity else ""
    
    return f"""## Task
Answer the following question based on the provided context.

## Instructions
1. Review the question and optional key entities
2. Apply the provided formatting rules
3. Extract the answer from the context

## Constraints
- MUST: Provide ONLY the JSON object
- MUST NOT: Add commentary outside the JSON

## Information
Question: {question} {q_key_entity}
{key_entity_text}

Formatting Rules:
{rules_text}

Context:
{context_str}

## Output Format (JSON)
{output_format}
"""


def get_answer_rules_prediction_system_prompt() -> str:
    
    return """You are an expert in understanding and question analysis.

Your task is to analyze questions and determine the best answer formatting rules to apply.

Consider:
- Question patterns (yes/no, comparison, choice, etc.)
- Entity types being asked about (person, place, biomedical, etc.)
- Special cases (royal/noble figures, compound answers, etc.)."""


def get_answer_rules_prediction_user_prompt(
    question: str,
    rules_reference: str,
    dataset: Optional[str] = None,
    type_q: Optional[str] = None
) -> str:
    """
    Build user prompt for answer rules prediction.
    
    Args:
        question: The question to analyze
        rules_reference: JSON string of available rules
        dataset: Dataset context (medhop, bioasq, cwq, wiki)
        type_q: Question type hint (inference, compositional, comparison, bridge_comparison)
        
    Returns:
        Complete user prompt string
    """
    context_info = ""
    if dataset:
        context_info += f"\nDataset: {dataset}"
        if dataset in ["medhop", "bioasq"]:
            context_info += " (biomedical/scientific domain)"
        elif dataset == "wiki":
            context_info += " (general knowledge, Wikipedia-based)"
        elif dataset == "cwq":
            context_info += " (complex web questions)"
    
    if type_q:
        context_info += f"\nQuestion Type Hint: {type_q}"
        if type_q == "inference":
            context_info += " (family relations, royal/noble figures)"
        elif type_q == "compositional":
            context_info += " (multi-hop facts about entities)"
        elif type_q in ["comparison", "bridge_comparison"]:
            context_info += " (comparing two entities)"

    return f"""Analyze this question and determine the appropriate answer formatting rules.

Question: "{question}"
{context_info}

Available Rules Reference:
{rules_reference}

Task: Analyze the question and output a JSON object with:
1. "question_type": The primary type of question (from the rules reference keys)
2. "primary_rule": The main rule name that should be applied
3. "secondary_rules": Array of additional rule names that may also apply
4. "formatting_instructions": Specific formatting instructions for this question
5. "examples": 2-3 example answers that would follow the rules
6. "reasoning": Brief explanation of why these rules were chosen

Output ONLY valid JSON, no additional text."""


# ============================================================================
# ANSWER SELECTION PROMPTS - Extracted from handle_multi_choice.py
# ============================================================================

def get_answer_selection_user_prompt(
    question: str,
    answers: List[Dict[str, Any]]
) -> str:
    """
    Build user prompt for answer selection from multiple candidates.
    
    Optimized for o3-mini: minimalist and constraint-focused.
    
    Args:
        question: The question being answered
        answers: List of answer dicts with 'short_answer' and 'long_answer'
        
    Returns:
        Complete user prompt string
    """
    # Format answers for comparison
    formatted_answers = "\n\n".join([
        f"[{idx + 1}] Short: {ans.get('short_answer', 'N/A')}\n    Long: {ans.get('long_answer', 'N/A')}"
        for idx, ans in enumerate(answers)
    ])
    
    return f"""## Task
Select the single best answer to the question from {len(answers)} candidates.

## Constraints
- MUST: Output valid JSON only
- MUST: Choose exactly one answer (index 1-{len(answers)})
- Selection criteria (in priority order):
  1. Accuracy: Answer correctly addresses the question
  2. Specificity: Answer is appropriately detailed/specific
  3. Evidence quality: Reasoning in long_answer is well-supported
  4. Consistency: Answer is internally consistent

## Information
Question: {question}

Candidates:
{formatted_answers}

## Output Format (JSON)
{{
    "answer_index": <1 to {len(answers)}>,
    "reasoning": "Brief explanation of why this answer is best",
    "confidence": "high|medium|low"
}}"""


def get_parallel_answers_combination_prompt(question: str, context: str) -> str:
    """
    Get prompt for combining parallel answers from multiple sources.
    
    Used in generate_combine_parallel_answers().
    
    Args:
        question: The question being answered
        context: Combined context from multiple sources
        
    Returns:
        Prompt string
    """
    return f"""## Role
You are an expert in logical synthesis

## Task
Analyze the provided information to determine the correct answer and reconcile sources

## Instructions
1. Analyze the question and context
2. Generate a valid JSON object with detailed reasoning (long_answer) and direct answer (short_answer)

## Rules
### Choice Questions
- Return the name of the chosen option extracted EXACTLY from the question

### Confirmation Questions
- Return "yes" or "no" (lowercase)

## Constraints
- MUST: Provide ONLY the JSON object
- MUST NOT: Include commentary outside the JSON

## Information
Question: {question}
Context: {context}

## Output Format (JSON)
{{
    "long_answer": "Reasoning synthesized from information",
    "short_answer": "The final answer following the rules"
}}
"""


def get_multiple_choice_selection_prompt(question: str, context: str) -> str:
    """
    Get prompt for selecting answer from multiple choice options.
    
    Used in gen_ans_multiple_choice().
    
    Args:
        question: The question being answered
        context: Context containing multiple choice options
        
    Returns:
        Prompt string
    """
    return f"""Question: {question}
Instructions:
- Please base your answer ONLY on the context provided
- Please only choose the given answers (without additional explanations). For example: "A", "B", "C", "D", "E", "F", "G"
- Do not include anything else aside from your answer

Context:
{context}
Please provide your short answer to the question based on the context above."""

COMPREHENSIVE_VALIDATION_RULES = {
    "question_type_detection": {
        "description": "Identify and match question type patterns",
        "categories": {
            "count": {
                "keywords": ["how many", "how much", "number of", "total"],
                "format": "ONLY a number or numeric phrase (e.g., '10', 'ten')",
                "rule": "No extra text allowed"
            },
            "time_duration": {
                "keywords": ["how long", "duration", "how many hours", "how many days"],
                "format": "Concise time value (e.g., '36 hours', '3 days')",
                "rule": "Must be time-specific"
            },
            "yes_no": {
                "keywords": ["is ", "are ", "do ", "does ", "can ", "could ", "will ", "would ", "should "],
                "format": "Exactly 'Yes' or 'No'",
                "rule": "Binary answer only"
            },
            "entity_biomedical": {
                "keywords": ["what", "who", "which"],
                "domain": "biomedical/scientific",
                "format": "FULL noun phrase (e.g., 'chromosome X', 'interleukin-6', 'dopamine receptor D2')",
                "rule": "Must be complete biomedical terminology"
            },
            "entity_general": {
                "keywords": ["what", "who", "which"],
                "domain": "general knowledge/people/films/places",
                "format": "Proper names are VALID (e.g., 'Manuel Romero', 'Paris')",
                "rule": "Context determines biomedical vs general"
            },
            "organism": {
                "keywords": ["what organism", "what type of organism"],
                "format": "Organism type (e.g., 'bacterium', 'fungus')",
                "rule": "Must be organism classification"
            },
            "range_interval": {
                "keywords": ["range", "between", "from", "to"],
                "format": "Numeric range (e.g., '300-330 msec', '10-20%')",
                "rule": "Must be interval format"
            },
            "comparison": {
                "keywords": ["older", "younger", "later", "earlier", "which", "who has"],
                "format": "NAME of chosen option extracted EXACTLY from question",
                "rule": "Return option name, NOT computed value (not a date for older/younger)"
            },
            "confirmation": {
                "keywords": ["are ", "is ", "same", "different", "both"],
                "format": "'yes' or 'no' (lowercase)",
                "rule": "Binary confirmation only"
            },
            "inference_royal_noble": {
                "keywords": ["father", "mother", "son", "daughter", "king", "queen", "prince", "princess"],
                "format": "prefix|name|suffix (e.g., 'King|William IV|')",
                "rule": "Include royal prefixes and hereditary titles, EXCLUDE territorial designations"
            },
            "compositional": {
                "keywords": ["where", "nationality", "education", "occupation"],
                "format": "Entity-specific (most specific location: city > state > country)",
                "rule": "Connect facts across multiple documents"
            }
        }
    },
    "medhop_biomedical_rules": {
        "description": "Rules specific to MedHop/BioASQ biomedical domain",
        "count_questions": {
            "rule": "Answer must be ONLY a number or numeric phrase",
            "examples": ["10", "ten", "twenty-five"],
            "violations": ["extra text", "non-numeric content"]
        },
        "time_duration_questions": {
            "rule": "Answer must be concise time value",
            "examples": ["36 hours", "3 days", "2 weeks"],
            "violations": ["imprecise duration", "narrative time references"]
        },
        "yes_no_questions": {
            "rule": "Answer must be exactly 'Yes' or 'No'",
            "violations": ["yes/no with explanation", "maybe", "uncertain"]
        },
        "entity_questions": {
            "rule": "Return FULL noun phrase for biomedical/scientific entities",
            "examples": ["chromosome X", "interleukin-6", "dopamine receptor D2"],
            "violations": ["partial names", "abbreviations without full form"]
        },
        "organism_questions": {
            "rule": "Return organism type/classification",
            "examples": ["bacterium", "fungus", "virus"],
            "violations": ["specific organism name instead of type"]
        },
        "range_questions": {
            "rule": "Return numeric range with units",
            "examples": ["300-330 msec", "10-20%"],
            "violations": ["single value instead of range"]
        }
    },
    "wiki_general_knowledge_rules": {
        "description": "Rules for Wikipedia/general knowledge domain",
        "comparison_questions": {
            "rule": "Answer must be the NAME of chosen option extracted EXACTLY from question",
            "examples": [
                "Question: 'Which film has director born later, A Flame In My Heart or Butcher, Baker?'",
                "Answer: 'A Flame In My Heart' (NOT a date)"
            ],
            "violations": ["computed value", "date instead of option name", "paraphrased option"]
        },
        "confirmation_questions": {
            "rule": "Answer must be 'yes' or 'no' (lowercase)",
            "examples": ["yes", "no"],
            "violations": ["Yes/No (capitalized)", "explanation after yes/no"]
        },
        "inference_royal_noble": {
            "rule": "Format: prefix|name|suffix with mandatory royal prefixes",
            "prefixes_required": ["King", "Queen", "Prince", "Princess", "Duke", "Duchess", "Sir", "Dame", "Lord", "Lady"],
            "include": ["numbered hereditary titles: '2nd Earl of Stafford'"],
            "exclude": ["territorial designations based only on land ownership"],
            "examples": ["King|William IV|", "Princess|Luisa Carlotta|of Naples and Sicily"],
            "violations": ["missing royal prefix", "parentheses in answer", "territorial designations"]
        },
        "compound_answers": {
            "rule": "Deconstruct multiple concepts, join with pipe '|' separator",
            "examples": ["Presidential|Unitary|Democratic", "Drawing|Painting|Photography"],
            "violations": ["unseparated concepts", "single compound noun"]
        }
    },
    "location_rules": {
        "description": "Rules for location/geography answers",
        "specificity": "Use MOST SPECIFIC (city > state > country)",
        "generation": "Do NOT generate location names not in context",
        "fallback": {
            "rule": "For birthplace/death without explicit mention: nationality → country as fallback",
            "must_mark": "Clearly mark as INFERRED if using nationality fallback"
        },
        "violations": ["too general location", "hallucinated place", "incorrect specificity"]
    },
    "nationality_rules": {
        "description": "Rules for nationality/citizenship answers",
        "patterns": {
            "x_born_y": {"pattern": "X-born Y", "answer": "X"},
            "x_y_nationality": {"pattern": "X-Y nationality", "answer": "X"},
            "x_y_space": {"pattern": "X Y nationality", "answer": "Y"}
        }
    },
    "content_validation": {
        "description": "General content validation rules",
        "requirements": [
            "No hallucinated entities/locations",
            "Synonyms/alternate names treated as same entity",
            "Evidence properly cited in long_answer",
            "Explicit evidence prioritized over inference",
            "'Insufficient information' is VALID answer when context truly lacks information"
        ],
        "long_answer_requirements": {
            "flexibility": "Accept any reasonable explanation connecting evidence to answer",
            "optional": ["citing document IDs", "step-by-step format", "rigid structure"],
            "priority": "Check that long_answer is consistent with short_answer and contains reasoning",
            "strict": False
        }
    },
    "insufficient_information": {
        "description": "Rules for 'Insufficient information' answers",
        "when_valid": "Context truly lacks needed information",
        "short_answer": "Insufficient information",
        "long_answer": "Should explain why information is unavailable, but be flexible with format",
        "content_check": "SKIP strict content checks for this answer type"
    }
}


def get_rules_for_question(question: str, dataset: Optional[str] = None, type_q: Optional[str] = None) -> Dict[str, Any]:
    """
    Get all applicable rules for a question without using if-else logic.
    
    This function analyzes the question and returns matching rules from COMPREHENSIVE_VALIDATION_RULES
    based on keyword matching and patterns, allowing for generalization across question types.
    
    Args:
        question: The question text to analyze
        dataset: Optional dataset context (medhop, bioasq, cwq, wiki)
        type_q: Optional question type hint (inference, compositional, comparison, bridge_comparison)
        
    Returns:
        Dict containing all relevant rules, question detection, and formatting requirements
    """
    question_lower = question.lower()
    applicable_rules = {}
    
    # Detect question type by keyword matching (no if-else)
    question_type_detection = COMPREHENSIVE_VALIDATION_RULES["question_type_detection"]
    matched_types = []
    
    for qtype, qtype_info in question_type_detection["categories"].items():
        keywords = qtype_info.get("keywords", [])
        if any(kw in question_lower for kw in keywords):
            matched_types.append({
                "type": qtype,
                "format": qtype_info.get("format", ""),
                "rule": qtype_info.get("rule", ""),
                "keywords_matched": [kw for kw in keywords if kw in question_lower]
            })
    
    applicable_rules["detected_question_types"] = matched_types
    
    # Apply domain-specific rules based on dataset
    if dataset in ["medhop", "bioasq"]:
        applicable_rules["domain"] = "biomedical"
        applicable_rules["domain_rules"] = COMPREHENSIVE_VALIDATION_RULES["medhop_biomedical_rules"]
    elif dataset in ["cwq", "wiki"]:
        applicable_rules["domain"] = "general_knowledge"
        applicable_rules["domain_rules"] = COMPREHENSIVE_VALIDATION_RULES["wiki_general_knowledge_rules"]
    else:
        applicable_rules["domain"] = "unknown"
    
    # Add universal rules
    applicable_rules["location_rules"] = COMPREHENSIVE_VALIDATION_RULES["location_rules"]
    applicable_rules["nationality_rules"] = COMPREHENSIVE_VALIDATION_RULES["nationality_rules"]
    applicable_rules["content_validation"] = COMPREHENSIVE_VALIDATION_RULES["content_validation"]
    applicable_rules["insufficient_information"] = COMPREHENSIVE_VALIDATION_RULES["insufficient_information"]
    
    # Add type-specific hint if provided
    if type_q:
        applicable_rules["type_q_hint"] = type_q
    
    return applicable_rules


def get_prompt_for_answer_validation(question: str, answer: dict, dataset: str = "", type_q: str = "") -> List[Dict[str, str]]:
    """
    Generate validation prompt for checking if answer follows normalization rules.
    Uses comprehensive validation checklist covering all question types and domains.
    NO IF-ELSE LOGIC - All rules embedded in prompt for LLM to detect and apply.
    
    Args:
        question: The question being answered
        answer: Dictionary with 'short_answer' and 'long_answer' keys
        dataset: Dataset name (optional, for context only - not used for conditional logic)
        type_q: Question type (optional, for context only - not used for conditional logic)
    
    Returns:
        List of message dictionaries for GPT API
    
    Example:
        >>> messages = get_prompt_for_answer_validation(
        ...     question="How many genes are involved?",
        ...     answer={"short_answer": "5", "long_answer": "Based on the evidence..."}
        ... )
        >>> response = gpt_request(messages, model="gpt-4o-mini")
    """
    system_message = """## Role
You are an expert answer validator for multi-hop question answering systems

## Task
Validate answers against comprehensive normalization rules covering biomedical, general knowledge, and specialized question types."""
    
    user_prompt = f"""## Task
Validate if this answer follows all normalization rules. Think step-by-step

## Input
Question: {question}
Short Answer: {answer.get('short_answer', '')}
Long Answer: {answer.get('long_answer', '')}

## Validation Checklist

### 1. Question Type Detection
- Identify question type: count ("how many"), time/duration, yes/no, entity (Who/Which/What), organism, comparison, inference, compositional
- Verify answer format matches detected type

### 2. Biomedical Rules
- Count Questions: Answer must be ONLY a number or numeric phrase (e.g., "10", "ten"). No extra text allowed
- Time/Duration Questions: Answer must be concise time value (e.g., "36 hours", "3 days")
- Yes/No Questions: Answer must be exactly "Yes" or "No"
- Entity Questions (Who/Which/What):
  - For BIOMEDICAL/SCIENTIFIC entities: Must return FULL noun phrase (e.g., "chromosome X", "interleukin-6")
  - For GENERAL KNOWLEDGE (people/films/places): Proper names are VALID (e.g., "Manuel Romero", "Paris")
- Organism Questions: Return organism type (e.g., "bacterium", "fungus")
- Range/Interval Questions: Return numeric range (e.g., "300-330 msec", "10-20%")
- Lists: If none of above apply, provide concise comma-separated items

### 3. Wiki/General Knowledge Rules
- Comparison Questions (Which/Who...older/younger/later/earlier): Answer must be the NAME of chosen option extracted EXACTLY from the question (NOT a date)
- Confirmation Questions (Are/Do...same): Answer must be "yes" or "no" (lowercase)
- Inference Questions (Royal/Noble Figures):
  - Royal prefix required if applicable: King/Queen, Prince/Princess, Duke/Duchess, etc
  - Include numbered hereditary titles: "2nd Earl of Stafford"
  - Include honorifics: Sir/Dame/Lord/Lady
  - EXCLUDE territorial designations based only on land ownership
  - NO parentheses in final answer
- Compound Answers: Multiple concepts MUST be deconstructed with pipe "|" separator

### 4. Location Rules
- Always use MOST SPECIFIC location (city > state > country)
- MUST NOT: Generate location names not in context
- For birthplace/death without explicit mention: use nationality → country as fallback
- Clearly mark as INFERRED if using nationality fallback

### 5. Nationality Rules
- "X-born Y" → answer "X"
- "X-Y nationality" → answer "X"
- "X Y nationality" → answer "Y"

### 6. Content Validation
- No hallucinated entities/locations
- Synonyms/alternate names treated as same entity
- Evidence properly cited in long_answer
- Explicit evidence prioritized over inference
- "Insufficient information" is a VALID answer when context truly lacks needed information

### 7. Special Case - "Insufficient information"
- Short answer = "Insufficient information" → SKIP strict content checks
- Long_answer should explain why information is unavailable, but be flexible with format

### 8. Long Answer Validation (Flexible)
- Long answer should generally support the short answer with reasoning
- Accept any reasonable explanation that connects evidence to the answer
- Citing document IDs or specific evidence sources is good but not required
- Step-by-step explanations are encouraged but not mandatory
- PRIORITY: Check only that long_answer is consistent with short_answer and contains some reasoning
- MUST NOT: Enforce rigid formatting requirements on the long answer

## Output Format (JSON)
{{
    "is_valid": true/false,
    "violations": ["list of specific rule violations found with rule category, empty if valid"],
    "additional_rules": "Specific corrective instructions to inject into generator prompt (only if is_valid=false, otherwise empty string)"
}}
"""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages
