"""Predict answer formatting rules from question using LLM."""

import json
from typing import Dict
from src.gpt_request import request_api


def predict_answer_rules(question: str, model: str = "gpt-4o-mini") -> Dict:
    
    system_msg = "You are an expert at analyzing questions and determining answer formatting requirements."
    
    user_msg = f"""Analyze this question and provide formatting rules.

Question: {question}

Return JSON with:
{{
  "formatting_instructions": "Brief formatting rules",
  "needs_normalization": true/false
}}"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    
    result = request_api(messages, model=model)
    
    if not result:
        return {"formatting_instructions": "Provide a clear, direct answer.", "needs_normalization": True}
    
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {"formatting_instructions": "Provide a clear, direct answer.", "needs_normalization": True}
