"""
Main entry points:
- generate_answer_new(): NEW PIPELINE (predict rules → generate → normalize → select)
- generate_answer(): LEGACY (kept for backward compatibility)

Helper functions also available:
- generate_answer(): Generate with pre-computed rules
- apply_answer_normalization_rules(): Get formatting rules for questions (returns primary + multiple secondary rules)
- select_best_answer(): Select from multiple candidate answers
- normalize_answer_with_wikipedia(): Normalize using Wikipedia API
"""

from src.prompt import (
    load_prompt_gen_ans,
    get_parallel_answers_combination_prompt,
    get_multiple_choice_selection_prompt,
    get_answer_generation_system_prompt,
    get_answer_generation_context_synthesis_rules,
    get_answer_generation_type_specific_rules,
    get_answer_generation_output_format,
    get_answer_generation_user_prompt,
    get_parallel_answers_combination_prompt_wiki,
    get_answer_rules_prediction_system_prompt,
    get_answer_rules_prediction_user_prompt
)
from src.gpt_request import *
import json
import time
from typing import List, Dict, Any, Optional, Union

def _generate_answer(question, search_kw, key_entities, relevant_info, dataset=None, type_q=None, target_answer_type=None, 
                     using_target_type=False, using_gen_old=False, has_extract_ner=False,using_prompt_yn=False, model="o3-mini"):
    prompt = load_prompt_gen_ans(question, search_kw, key_entities, relevant_info, dataset=dataset, type_q=type_q, target_answer_type=target_answer_type, 
                                 using_target_type=using_target_type, using_gen_old=using_gen_old, has_extract_ner=has_extract_ner, using_prompt_yn=using_prompt_yn)
    if dataset in ["cwq", "wiki"]:
        messages = [
            {"role": "system", "content": "## Role\nYou are an expert in general knowledge.\n\n## Task\nProvide accurate answers based on the provided information."},
            {"role": "user", "content": prompt}
        ]
    else:
        messages=[
            {"role": "system", "content": "## Role\nYou are an expert in medical and scientific knowledge.\n\n## Task\nProvide accurate answers based on the provided information."},
            {"role": "user", "content": prompt}
        ]
    
    result_json = request_api(messages, model)
    
    #call openai
    # result_json = request_api_o3_mini(messages)
    
    # try:
    #     result_dict = json.loads(result_json)
    # except Exception as e:
    #     print(f"Error parsing JSON response: {e}")
    #     print(f"Raw response: {result_json}")

    #     long_answer = ""
    #     short_answer = ""
        
    #     if "long_answer" in result_json.lower() and "short_answer" in result_json.lower():
    #         parts = result_json.split("long_answer")
    #         if len(parts) > 1:
    #             remaining = parts[1].strip()
    #             if "short_answer" in remaining:
    #                 answers = remaining.split("short_answer")
    #                 long_answer = answers[0].strip().strip('":').strip()
    #                 short_answer = answers[1].strip().strip('":').strip()
    #     result_dict = {
    #         "long_answer": long_answer,
    #         "short_answer": short_answer
    #     }
    
    clean_json = result_json.replace("```json", "").replace("```", "").strip()

    try:
        result_dict = json.loads(clean_json)
    except Exception as e:
        print(f"JSON vẫn lỗi: {e}. Đang bóc tách thủ công...")
        
        long_answer = ""
        short_answer = ""
        
        if '"short_answer":' in clean_json:
            tmp = clean_json.split('"short_answer":')[1]
            short_answer = tmp.split(',')[0].replace('"', '').replace(':', '').strip()
            
        if '"long_answer":' in clean_json:
            tmp = clean_json.split('"long_answer":')[1]
            long_answer = tmp.rsplit('}', 1)[0].replace('"', '').replace(':', '').strip()

        result_dict = {
            "long_answer": long_answer,
            "short_answer": short_answer
        }
 
    return result_dict

def generate_combine_parallel_answers(question, context, dataset='wiki', model="o3-mini"):
    if dataset == 'medhop':
        prompt = get_parallel_answers_combination_prompt(question, context)
    else:
        prompt = get_parallel_answers_combination_prompt_wiki(question, context)
        
    messages = [
        {"role": "system", "content": "You are an expert in general knowledge. Your task is to provide accurate answers based on the provided information. Please only output the choice (without additional explanations)"},
        {"role": "user", "content": prompt}
    ]
    
    result_json = request_api(messages, model)
    
    try:
        result_dict = json.loads(result_json)
        long_answer = result_dict.get("long_answer", "")
        short_answer = result_dict.get("short_answer", "")
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {result_json}")

        long_answer = ""
        short_answer = ""
        
        if "long_answer" in result_json.lower() and "short_answer" in result_json.lower():
            parts = result_json.split("long_answer")
            if len(parts) > 1:
                remaining = parts[1].strip()
                if "short_answer" in remaining:
                    answers = remaining.split("short_answer")
                    long_answer = answers[0].strip().strip('":').strip()
                    short_answer = answers[1].strip().strip('":').strip()
        
        if not long_answer:
            long_answer = result_json
            short_answer = "No relevant answer found"
    
    return {
        "long_answer": long_answer,
        "short_answer": short_answer
    }

def gen_ans_multiple_choice(question, context, dataset='medhop', model = "o3-mini"):
    prompt = get_multiple_choice_selection_prompt(question, context)
    if dataset == 'wiki':
        messages = [
            {"role": "system", "content": "## Role\nYou are an expert in general knowledge.\n\n## Task\nSelect the correct choice from the provided information.\n\n## Constraints\n- MUST: Output ONLY the choice (e.g., 'A')\n- MUST NOT: Include any additional explanations"},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "system", "content": "## Role\nYou are an expert in medical and scientific knowledge.\n\n## Task\nSelect the correct choice from the provided information.\n\n## Constraints\n- MUST: Output ONLY the choice (e.g., 'A')\n- MUST NOT: Include any additional explanations"},
            {"role": "user", "content": prompt}
        ]
    
    result_json = request_api(messages, model)
    
    return result_json

ANSWER_RULES = {
   "count": {
       "description": "Questions asking 'how many' or about counts",
       "format": "Return ONLY a number or numeric phrase (e.g., '10', 'ten')",
       "examples": ["10", "ten", "three hundred"]
   },
   "range": {
       "description": "Questions asking for a range or estimated interval",
       "format": "Return the numeric range (e.g., '300-330 msec', '10-20%')",
       "examples": ["300-330 msec", "10-20%", "5-10 days"]
   },
   "time_duration": {
       "description": "Questions asking about time or duration",
       "format": "Return a concise time value (e.g., '36 hours', '3 days')",
       "examples": ["36 hours", "3 days", "2 weeks"]
   },
   "yes_no": {
       "description": "Yes/no questions",
       "format": "Return exactly 'Yes' or 'No'",
       "examples": ["Yes", "No"]
   },
   "biomedical_entity": {
       "description": "Questions about biomedical/scientific entities (genes, proteins, diseases, etc.)",
       "format": "Return ONLY the SINGLE most probable entity as a FULL noun phrase. NO lists or separators. MAX 3 words.",
       "examples": ["chromosome X", "interleukin-6", "dopamine receptor"]
   },
   "general_entity": {
       "description": "Questions about general entities (people, places, films, etc.)",
       "format": "Return the proper name directly",
       "examples": ["Manuel Romero", "Paris", "The Godfather"]
   },
   "organism": {
       "description": "Questions asking about organism type",
       "format": "Return the organism type",
       "examples": ["bacterium", "fungus", "virus"]
   },
   "comparison_choice": {
       "description": "Comparison questions asking to choose (Which/Who...older/younger/later/earlier)",
       "format": "Return the NAME of the chosen option extracted EXACTLY from the question",
       "examples": ["A Flame In My Heart", "Butcher, Baker, Nightmare Maker"]
   },
   "compound": {
       "description": "Answers with multiple concepts",
       "format": "Select ONLY the SINGLE most probable answer. DO NOT use list or separator. MAX 3 words.",
       "examples": ["Presidential", "Ectodermal dysplasia", "HLA-B51", "Interleukin-6"]
   },
   "location": {
       "description": "Questions asking about locations",
       "format": "Return the MOST SPECIFIC location (city > state > country). Do NOT generate locations not in context",
       "examples": ["Valoria", "Silver Creek", "London"]
   },
   "nationality": {
       "description": "Questions about nationality",
       "format": "'X-born Y' → 'X'; 'X-Y nationality' → 'X'; 'X Y nationality' → 'Y'",
       "examples": ["Spanish", "Mexican", "German"]
   },
   "default": {
       "description": "Default rule for other questions",
       "format": "Select ONLY the SINGLE most probable answer. NO lists, NO separators. MAX 3 words.",
       "examples": []
   },
   "insufficient_info": {
       "description": "When information is not available in context",
       "format": "Return exactly 'Insufficient information'",
       "examples": ["Insufficient information"]
   }
}


def apply_answer_normalization_rules(
    question: str,
    dataset: Optional[str] = None,
    type_q: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0
) -> Dict:
    """
    Predict the answer formatting rules based on question analysis.
    
    Args:
        question: The question to analyze
        dataset: Dataset context (medhop, bioasq, cwq, wiki)
        type_q: Question type (inference, compositional, comparison, bridge_comparison)
        model: LLM model to use
        temperature: Temperature for generation
    
    Returns:
        Dict containing:
        - question_type: Identified question type
        - primary_rule: Main formatting rule to apply
        - secondary_rules: List of additional rules that may apply (can be multiple)
        - formatting_instructions: Specific instructions
        - examples: Example answers
        - needs_normalization: Boolean indicating if answer should be normalized
    """
    system_message = get_answer_rules_prediction_system_prompt()
    rules_reference = json.dumps(ANSWER_RULES, indent=2)
    user_prompt = get_answer_rules_prediction_user_prompt(question, rules_reference, dataset, type_q)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    
    result = request_api(messages=messages, model=model, temperature=int(temperature))
    
    if not result or not result.strip():
        return get_default_rules(question, dataset, type_q)
    
    try:
        parsed_result = json.loads(result)
        
        # Enrich with rule details
        primary_rule_name = parsed_result.get("primary_rule", "default")
        if primary_rule_name in ANSWER_RULES:
            parsed_result["primary_rule_details"] = ANSWER_RULES[primary_rule_name]
        
        secondary_rules = parsed_result.get("secondary_rules", [])
        parsed_result["secondary_rule_details"] = [
            ANSWER_RULES[rule] for rule in secondary_rules if rule in ANSWER_RULES
        ]
        
        # Determine if normalization is needed
        parsed_result["needs_normalization"] = should_normalize(primary_rule_name, secondary_rules)
        
        return parsed_result
        
    except json.JSONDecodeError:
        return get_default_rules(question, dataset, type_q)


def should_normalize(primary_rule: str, secondary_rules: List[str]) -> bool:
    """Determine if the answer should be normalized with Wikipedia."""
    non_normalizable_rules = {
        "count", "range", "time_duration", "yes_no", "confirmation", "insufficient_info"
    }
    return primary_rule not in non_normalizable_rules


def get_default_rules(question: str, dataset: Optional[str], type_q: Optional[str]) -> Dict:
    """Generate default rules based on simple heuristics when LLM fails."""
    question_lower = question.lower()
    
    if "how many" in question_lower:
        primary_rule = "count"
    elif question_lower.startswith(("is ", "are ", "do ", "does ", "did ", "was ", "were ", "can ", "could ")):
        if type_q in ["comparison", "bridge_comparison"]:
            primary_rule = "confirmation"
        else:
            primary_rule = "yes_no"
    elif "which" in question_lower and any(word in question_lower for word in ["older", "younger", "earlier", "later", "first", "last"]):
        primary_rule = "comparison_choice"
    elif type_q == "inference":
        primary_rule = "royal_noble"
    elif any(word in question_lower for word in ["where", "country", "city", "location", "place"]):
        primary_rule = "location"
    elif any(word in question_lower for word in ["nationality", "citizen"]):
        primary_rule = "nationality"
    elif dataset in ["medhop", "bioasq"]:
        primary_rule = "biomedical_entity"
    else:
        primary_rule = "general_entity"
    
    return {
        "question_type": primary_rule,
        "primary_rule": primary_rule,
        "secondary_rules": ["insufficient_info"],
        "formatting_instructions": ANSWER_RULES.get(primary_rule, ANSWER_RULES["default"])["format"],
        "examples": ANSWER_RULES.get(primary_rule, ANSWER_RULES["default"])["examples"],
        "reasoning": "Default heuristic-based rule detection",
        "primary_rule_details": ANSWER_RULES.get(primary_rule, ANSWER_RULES["default"]),
        "secondary_rule_details": [ANSWER_RULES["insufficient_info"]],
        "needs_normalization": should_normalize(primary_rule, ["insufficient_info"])
    }


def get_rules_for_prompt(rules: Dict) -> str:
    """Convert rules dict to a formatted string for injection into answer generation prompt."""
    primary_details = rules.get("primary_rule_details", {})
    formatting = rules.get("formatting_instructions", "")
    examples = rules.get("examples", [])
    
    rules_text = f"""
### ANSWER FORMATTING RULES ###
Question Type: {rules.get('question_type', 'unknown')}

Primary Rule: {rules.get('primary_rule', 'default')}
Description: {primary_details.get('description', '')}
Format: {formatting}
"""
    
    if examples:
        rules_text += f"Examples: {', '.join(str(e) for e in examples[:3])}\n"
    
    secondary_rules = rules.get("secondary_rule_details", [])
    if secondary_rules:
        rules_text += "\nAdditional Rules:\n"
        for rule in secondary_rules:
            rules_text += f"- {rule.get('description', '')}: {rule.get('format', '')}\n"
    
    return rules_text

def generate_answer(
    question: str,
    target_answer_type: str,
    context: Union[str, List[str]],
    rules: Dict,
    q_key_entity: Optional[str] = None,
    evidence_key_entity: Optional[str] = None,
    dataset: Optional[str] = None,
    type_q: Optional[str] = None,
    additional_instructions: str = "",
    model: str = "gpt-4o-mini",
    temperature: float = 0
) -> Dict[str, str]:
    """
    Generate an answer using the provided rules for formatting.
    
    Args:
        question: The question to answer
        context: Context information (string or list of strings)
        rules: Rules dictionary from apply_answer_normalization_rules
        key_entity: Optional key entity being asked about
        dataset: Dataset context (medhop, bioasq, cwq, wiki)
        type_q: Question type hint
        additional_instructions: Extra instructions to inject
        model: LLM model to use
        temperature: Temperature for generation
        
    Returns:
        Dict with 'long_answer', 'short_answer', and optionally 'prefix', 'name', 'suffix'
    """
    # Handle context as list or string
    if isinstance(context, list):
        context_str = "\n\n".join(context)
    else:
        context_str = context
    
    # Get formatted rules for prompt
    rules_text = get_rules_for_prompt(rules)
    
    # Build system message
    system_message = get_answer_generation_system_prompt(dataset)
    
    # Build base rules section
    base_rules = get_answer_generation_context_synthesis_rules()
    
    # Add type-specific instructions
    type_specific = get_answer_generation_type_specific_rules(type_q, dataset)
    
    # Combine all instructions
    full_instructions = base_rules + type_specific
    
    if additional_instructions:
        full_instructions += f"\n### ADDITIONAL REQUIREMENTS ###\n{additional_instructions}\n"
    
    full_instructions += rules_text
    
    # Build output format
    output_format = get_answer_generation_output_format(type_q, rules.get("primary_rule") == "royal_noble", target_answer_type)
    
    # Build user prompt
    user_prompt = get_answer_generation_user_prompt(question, target_answer_type, context_str, full_instructions, output_format, q_key_entity, evidence_key_entity)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    
    result = request_api(messages=messages, model=model, temperature=int(temperature))
    
    if not result or not result.strip():
        return {"long_answer": "Unable to generate answer due to API error.", "short_answer": "Insufficient information"}
    
    try:
        answer = json.loads(result)
        if "short_answer" not in answer:
            answer["short_answer"] = "Insufficient information"
        if "long_answer" not in answer:
            answer["long_answer"] = "Unable to generate explanation."
        return answer
    except json.JSONDecodeError:
        return parse_fallback_answer(result)


def parse_fallback_answer(result: str) -> Dict[str, str]:
    """Try to parse answer from malformed response."""
    long_answer = ""
    short_answer = ""
    
    if "long_answer" in result.lower() and "short_answer" in result.lower():
        try:
            parts = result.split("long_answer")
            if len(parts) > 1:
                remaining = parts[1].strip()
                if "short_answer" in remaining:
                    answers = remaining.split("short_answer")
                    long_answer = answers[0].strip().strip('":,').strip()
                    short_answer = answers[1].strip().strip('":,').strip().rstrip('}').strip().strip('"')
        except Exception:
            pass
    
    if not long_answer:
        long_answer = "Unable to parse generated answer."
    if not short_answer:
        short_answer = "Insufficient information"
    
    return {"long_answer": long_answer, "short_answer": short_answer}

def normalize_answer_with_wikipedia(
    answer: Dict[str, str],
    should_normalize: bool = True
) -> Dict[str, Any]:
    """
    Normalize answer using Wikipedia API search to get canonical page titles.
    
    Args:
        answer: Dict with 'short_answer' and 'long_answer'
        should_normalize: Whether to normalize (False for numeric/boolean)
        
    Returns:
        Answer dict with normalized values and normalization metadata
    """
    from utils.norm_ans_with_wiki import get_title_1, get_title_2
    
    result = answer.copy()
    normalization_metadata = {
        "was_normalized": False,
        "original_short": answer.get("short_answer", ""),
        "changes": []
    }
    
    if not should_normalize:
        normalization_metadata["skip_reason"] = "Answer type does not require normalization"
        result["normalization"] = normalization_metadata
        return result
    
    short_ans = answer.get("short_answer", "")
    
    if not short_ans or short_ans in ["Insufficient information", "No answer"]:
        normalization_metadata["skip_reason"] = "No valid answer to normalize"
        result["normalization"] = normalization_metadata
        return result
    
    def get_wikipedia_title(entity_name: str) -> Optional[str]:
        """Get canonical Wikipedia page title."""
        if not entity_name or entity_name in ["Insufficient information", "No answer", "NAN"]:
            return None
        try:
            title = get_title_1(entity_name)
            if title:
                return title
            title = get_title_2(entity_name)
            if title:
                return title
            return None
        except Exception as e:
            print(f"Warning: Failed to get Wikipedia title for '{entity_name}': {e}")
            return None
    
    # Handle pipe-separated answers
    if "|" in short_ans:
        parts = short_ans.split("|")
        
        # Royal/noble format: prefix|name|suffix
        if len(parts) == 3:
            prefix, name, suffix = parts
            if name and name.upper() != "NAN":
                try:
                    normalized_name = get_wikipedia_title(name)
                    if normalized_name and normalized_name != name:
                        result["short_answer"] = f"{prefix}|{normalized_name}|{suffix}"
                        normalization_metadata["was_normalized"] = True
                        normalization_metadata["changes"].append({
                            "field": "name",
                            "original": name,
                            "normalized": normalized_name
                        })
                        if "name" in result:
                            result["name"] = normalized_name
                except Exception as e:
                    print(f"Warning: Failed to normalize '{name}': {e}")
        else:
            # Compound answer
            normalized_parts = []
            for idx, part in enumerate(parts):
                part = part.strip()
                if part:
                    try:
                        normalized_part = get_wikipedia_title(part)
                        if normalized_part and normalized_part != part:
                            normalized_parts.append(normalized_part)
                            normalization_metadata["changes"].append({
                                "field": f"part_{idx}",
                                "original": part,
                                "normalized": normalized_part
                            })
                        else:
                            normalized_parts.append(part)
                    except Exception:
                        normalized_parts.append(part)
                else:
                    normalized_parts.append(part)
            
            if normalization_metadata["changes"]:
                result["short_answer"] = "|".join(normalized_parts)
                normalization_metadata["was_normalized"] = True
    else:
        # Simple answer
        try:
            normalized_ans = get_wikipedia_title(short_ans)
            if normalized_ans and normalized_ans != short_ans:
                result["short_answer"] = normalized_ans
                normalization_metadata["was_normalized"] = True
                normalization_metadata["changes"].append({
                    "field": "short_answer",
                    "original": short_ans,
                    "normalized": normalized_ans
                })
        except Exception as e:
            print(f"Warning: Failed to normalize '{short_ans}': {e}")
    
    result["normalization"] = normalization_metadata
    
    # Add delay to avoid rate limiting
    if normalization_metadata["was_normalized"]:
        time.sleep(0.5)
    
    return result
