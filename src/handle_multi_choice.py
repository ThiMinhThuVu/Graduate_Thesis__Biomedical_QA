"""Select best answer from multiple candidates using LLM."""

import json
from typing import List, Dict, Any
from src.gpt_request import request_api


def select_best_answer(question: str, answers: List[Dict[str, Any]], model: str = "o3-mini") -> Dict[str, Any]:
    
    if not answers:
        return {"chosen_answer": {"short_answer": "No answer", "long_answer": "No answers provided"}, 
                "answer_index": 0, "reasoning": "No candidates", "confidence": "low"}
    
    if len(answers) == 1:
        return {"chosen_answer": answers[0], "answer_index": 1, 
                "reasoning": "Only one candidate", "confidence": "medium"}
    
    # Build comparison prompt
    candidates = "\n".join([f"Answer {i+1}: {a.get('short_answer', '')}" for i, a in enumerate(answers)])
    
    user_msg = f"""Select the best answer for this question.

Question: {question}

{candidates}

Return JSON with:
{{"answer_index": <number 1-{len(answers)}>, "reasoning": "why"}}"""

    result = request_api([{"role": "user", "content": user_msg}], model=model)
    
    if not result:
        return {"chosen_answer": answers[0], "answer_index": 1, 
                "reasoning": "API error", "confidence": "low"}
    
    try:
        parsed = json.loads(result)
        idx = parsed.get("answer_index", 1)
        if not isinstance(idx, int) or idx < 1 or idx > len(answers):
            idx = 1
        return {"chosen_answer": answers[idx - 1], "answer_index": idx,
                "reasoning": parsed.get("reasoning", "Selected by model"), "confidence": "medium"}
    except json.JSONDecodeError:
        return {"chosen_answer": answers[0], "answer_index": 1, 
                "reasoning": "Parse error", "confidence": "low"}
