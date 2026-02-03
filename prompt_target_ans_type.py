from typing import Optional, List, Dict, Any
import json

def load_target_answer_type_prompt(question: str) -> list:
    """
    Generate a prompt for extracting target_answer_type from a question.
    
    Args:
        question: The question string to analyze
        
    Returns:
        List of messages for LLM to extract target_answer_type
    """
    messages = [
        {
            "role": "system",
            "content": """## Role
You are an expert in biomedical question analysis and answer type classification

## Task
Identify the target_answer_type for the given question

## Instructions
Determine the target_answer_type as follows:
1. OVERRIDING RULE: If question can be answered with Yes/No → target_answer_type MUST be "correctness"
2. If "How many"/"How much" → target_answer_type MUST be "number of [Entity]"
3. If asking about time/when → use "date" for specific date, OR "year" for year only
4. If "Who" question (asking about a person) → target_answer_type MUST be "person"
5. If CHOICE question (question contains explicit options like "A or B") → target_answer_type MUST be "choice"
6. If asking about ANATOMICAL STRUCTURE (muscle, bone, organ, nerve, etc.) → use the specific anatomy type asked (e.g., "muscle", "nerve", "chromosome")
7. OTHERWISE: Extract the specific noun phrase from the question that describes the answer type (e.g., "What is the name of the condition..." → "condition"). DO NOT output the specific words "specific noun phrase".
8. If none apply → "Undefined"

## Rules
- Extract the most precise answer type that describes what the question is asking for
- Preserve important modifiers that specify the type (e.g., "gene mutation" not just "gene")
- For questions asking about properties, use the property name (e.g., "year", "suburb")
- For anatomy questions, extract the anatomical structure type being asked (e.g., "Which thick triangular muscle..." → "muscle")
- For choice questions, the answer will be one of the options provided in the question

## Constraints
- MUST: Output exactly one target_answer_type
- MUST: Follow the priority order: Yes/No > Quantitative > Temporal > Person > Choice > Anatomy > Specific extracted noun phrase

## Output Format (JSON)
{
    "target_answer_type": "the identified answer type"
}
"""
        },
        {
            "role": "user",
            "content": f"""## Examples (6)

---
**Example 1:**
Question: "Is aspirin effective for treating headaches?"

JSON desired output format:
{{
    "target_answer_type": "correctness"
}}
---
**Example 2:**
Question: "How many chromosomes do humans have?"

JSON desired output format:
{{
    "target_answer_type": "number of chromosomes"
}}
---
**Example 3:**
Question: "Which lactate dehydrogenase tetramer is most common in the brain?"

JSON desired output format:
{{
    "target_answer_type": "tetramer"
}}
---
**Example 4:**
Question: "In what year was penicillin discovered?"

JSON desired output format:
{{
    "target_answer_type": "year"
}}
---
**Example 5:**
Question: "Who discovered ring chromosomes in 1926?"

JSON desired output format:
{{
    "target_answer_type": "person"
}}
---
**Example 6:**
Question: "Which thick triangular muscle is most likely affected by Parsonage-Turner syndrome?"

JSON desired output format:
{{
    "target_answer_type": "muscle"
}}
---

## Information
Question: "{question}"

## Task
Identify the target_answer_type for the provided question. Output as JSON
"""
        }
    ]
    return messages

def load_batch_target_answer_type_prompt(questions: Dict[str, str]) -> list:
    """
    Generate a prompt for extracting target_answer_type from a batch of questions.
    
    Args:
        questions: Dictionary mapping sample_id to question string
        
    Returns:
        List of messages for LLM
    """
    # Format questions for the prompt
    questions_str = ""
    for q_id, q_text in questions.items():
        questions_str += f"ID: {q_id}\nQuestion: {q_text}\n\n"

    messages = [
        {
            "role": "system",
            "content": """## Role
You are an expert in biomedical question analysis and answer type classification

## Task
Identify the target_answer_type for each provided question in the batch.

## Instructions
Determine the target_answer_type as follows:
1. OVERRIDING RULE: If question can be answered with Yes/No → target_answer_type MUST be "correctness"
2. If "How many"/"How much" → target_answer_type MUST be "number of [Entity]"
3. If asking about time/when → use "date" for specific date, OR "year" for year only
4. If "Who" question (asking about a person) → target_answer_type MUST be "person"
5. If CHOICE question (question contains explicit options like "A or B") → target_answer_type MUST be "choice"
6. If asking about ANATOMICAL STRUCTURE (muscle, bone, organ, nerve, etc.) → use the specific anatomy type asked (e.g., "muscle", "nerve", "chromosome")
7. OTHERWISE: Extract the specific noun phrase from the question that describes the answer type (e.g., "What is the name of the condition..." → "condition"). DO NOT output the specific words "specific noun phrase".
8. If none apply → "Undefined"

## Rules
- Extract the most precise answer type that describes what the question is asking for
- Preserve important modifiers that specify the type (e.g., "gene mutation" not just "gene")
- For questions asking about properties, use the property name (e.g., "year", "suburb")
- For anatomy questions, extract the anatomical structure type being asked (e.g., "Which thick triangular muscle..." → "muscle")
- For choice questions, the answer will be one of the options provided in the question

## Constraints
- MUST: Output a JSON list of objects
- MUST: Include "id" and "target_answer_type" for EACH question
- MUST: Follow the priority order: Yes/No > Quantitative > Temporal > Person > Choice > Anatomy > Specific extracted noun phrase

## Output Format (JSON)
[
    {
        "id": "question_id_1",
        "target_answer_type": "identified answer type"
    },
    ...
]
"""
        },
        {
            "role": "user",
            "content": f"""## Examples
ID: ex1
Question: "Is aspirin effective for treating headaches?"

ID: ex2
Question: "How many chromosomes do humans have?"

ID: ex3
Question: "Which thick triangular muscle is most likely affected by Parsonage-Turner syndrome?"

JSON desired output format:
[
    {{
        "id": "ex1",
        "target_answer_type": "correctness"
    }},
    {{
        "id": "ex2",
        "target_answer_type": "number of chromosomes"
    }},
    {{
        "id": "ex3",
        "target_answer_type": "muscle"
    }}
]

## Batch Information
{questions_str}
## Task
Identify the target_answer_type for all questions above. Output as a JSON list.
"""
        }
    ]
    return messages