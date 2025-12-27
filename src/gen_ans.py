"""Generate answers from context with LLM-predicted formatting rules."""

import json
from typing import Dict, Optional, Union, List
from src.gpt_request import request_api
from src.predict_ans_rules import predict_answer_rules


def generate_answer_with_rules(
    question: str,
    context: Union[str, list],
    rules: Dict,
    model: str = "gpt-4o-mini",
    normalize: bool = False
) -> Dict:
    """Generate answer and return full history format."""
    
    context_str = "\n\n".join(context) if isinstance(context, list) else context
    
    system_msg = "You are an expert at answering questions accurately based on provided context."
    
    user_msg = f"""Answer this question using the context provided.

Question: {question}

Context:
{context_str}

Rules:
{rules.get('formatting_instructions', 'Provide a clear, direct answer.')}

Return JSON with:
{{"long_answer": "detailed explanation", "short_answer": "concise answer"}}"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    
    result = request_api(messages, model=model)
    
    if not result:
        answer = {"long_answer": "API error", "short_answer": "Insufficient information"}
    else:
        try:
            answer = json.loads(result)
            answer.setdefault("short_answer", "Insufficient information")
            answer.setdefault("long_answer", "Unable to generate explanation.")
        except json.JSONDecodeError:
            answer = _parse_fallback(result)
    
    # Build full format like ans.json
    return {
        "before_norm": answer.copy(),
        "after_norm": answer.copy(),  # Normalization done externally
        "context_used": context_str,
        "rules_applied": rules.get('formatting_instructions', 'default'),
        "model_used": model
    }


def _parse_fallback(result: str) -> Dict[str, str]:
    
    long_ans = short_ans = ""
    
    if "long_answer" in result.lower() and "short_answer" in result.lower():
        try:
            parts = result.split("long_answer", 1)[1].split("short_answer", 1)
            long_ans = parts[0].strip().strip('":,').strip()
            short_ans = parts[1].strip().strip('":,}').strip('"')
        except:
            pass
    
    return {
        "long_answer": long_ans or "Unable to parse answer.",
        "short_answer": short_ans or "Insufficient information"
    }


def generate_answer(
    question: str,
    context: Union[str, list],
    model: str = "gpt-4o-mini",
    normalize: bool = False,
    candidates: Optional[List] = None
) -> Dict:
    """Main entry - returns ans.json format with full history."""
    
    # Predict rules
    rules = predict_answer_rules(question, model=model)
    
    # Generate single answer or multiple candidates
    if candidates is None:
        result = generate_answer_with_rules(question, context, rules, model)
        
        # Apply normalization if needed
        if normalize and rules.get('needs_normalization', False):
            from src.generate_answer import normalize_answer_with_wikipedia
            result["after_norm"] = normalize_answer_with_wikipedia(result["before_norm"])
        
        # Return single-answer format
        return {
            "chosen_answer": result["after_norm"],
            "reasoning": f"Generated using {model} with rules: {rules.get('formatting_instructions', 'default')}",
            "answer_index": 0,
            "answer_options": [result]
        }
    
    # Multi-candidate selection
    from src.generate_answer import normalize_answer_with_wikipedia, select_best_answer
    
    options = []
    for ctx in candidates:
        opt = generate_answer_with_rules(question, ctx, rules, model)
        if normalize and rules.get('needs_normalization', False):
            opt["after_norm"] = normalize_answer_with_wikipedia(opt["before_norm"])
        options.append(opt)
    
    # Select best
    selection = select_best_answer(question, [opt["after_norm"] for opt in options])
    
    return {
        "chosen_answer": options[selection["answer_index"]]["after_norm"],
        "reasoning": selection["reasoning"],
        "answer_index": selection["answer_index"],
        "answer_options": options
    }
