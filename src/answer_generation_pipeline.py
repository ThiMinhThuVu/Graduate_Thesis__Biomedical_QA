"""
Pipeline Module

Full answer generation pipeline:
1. predict_ans_rules: Determine answer formatting rules + normalization status
2. gen_ans: Generate answer with injected rules
3. norm_ans: Normalize answer using Wikipedia search
4. handle_multi_choice: Select best answer from multiple contexts

Each context is processed individually through the pipeline to produce a single answer.
Multiple contexts produce multiple answers, which are then aggregated.
"""

import json
import time
from typing import List, Dict, Any, Optional, Union
from src.predict_ans_rules import predict_answer_rules
from src.gen_ans import generate_answer_with_rules
from src.handle_multi_choice import select_best_answer
from utils.norm_ans_with_wiki import get_title_1, get_title_2


def get_wikipedia_title(entity_name: str) -> Optional[str]:
    """
    Get the canonical Wikipedia page title for an entity name.
    Uses Wikipedia API search to find the correct page title.
    
    Args:
        entity_name: The entity name to search for
        
    Returns:
        The canonical Wikipedia page title, or None if not found
    """
    if not entity_name or entity_name in ["Insufficient information", "No answer", "NAN"]:
        return None
    
    try:
        # Try the wikipediaapi approach first (more direct)
        title = get_title_1(entity_name)
        if title:
            return title
        
        # Fallback to the OpenSearch API approach
        title = get_title_2(entity_name)
        if title:
            return title
        
        return None
    except Exception as e:
        print(f"Warning: Failed to get Wikipedia title for '{entity_name}': {e}")
        return None


def normalize_answer_with_wikipedia(
    answer: Dict[str, Any],
    should_normalize: bool = True
) -> Dict[str, Any]:
    """
    Normalize answer using Wikipedia search to get canonical page titles.
    Uses Wikipedia API to find the correct page title for entity names.
    Only normalizes if should_normalize is True (from predict_ans_rules).
    
    Args:
        answer: Dict with 'short_answer' and 'long_answer' keys
        should_normalize: Whether to normalize (False for numeric/boolean answers)
        
    Returns:
        Answer dict with normalized values and normalization metadata
    """
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
    
    # Handle pipe-separated answers (compound answers or royal/noble format)
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
                        # Also update the name field if present
                        if "name" in result:
                            result["name"] = normalized_name
                except Exception as e:
                    print(f"Warning: Failed to normalize '{name}': {e}")
        else:
            # Compound answer: normalize each part
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
                    except Exception as e:
                        print(f"Warning: Failed to normalize '{part}': {e}")
                        normalized_parts.append(part)
                else:
                    normalized_parts.append(part)
            
            if normalization_metadata["changes"]:
                result["short_answer"] = "|".join(normalized_parts)
                normalization_metadata["was_normalized"] = True
    else:
        # Simple answer: normalize directly using Wikipedia search
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
    
    # Add a small delay to avoid rate limiting
    if normalization_metadata["was_normalized"]:
        time.sleep(0.5)
    
    return result


def ans_for_single_context(
    question: str,
    context: Union[str, List[str]],
    key_entity: Optional[str] = None,
    dataset: Optional[str] = None,
    type_q: Optional[str] = None,
    rules: Optional[Dict[str, Any]] = None,
    normalize: bool = True,
    model: str = "gpt-4o-mini",
    temperature: float = 0
) -> Dict[str, Any]:
    """
    Process a single context through the full pipeline.
    
    Pipeline: predict_ans_rules → gen_ans → norm_ans (with Wikipedia search)
    
    Args:
        question: The question to answer
        context: Context string or list of context strings
        key_entity: Optional key entity being asked about
        dataset: Dataset context (medhop, bioasq, cwq, wiki)
        type_q: Question type hint (inference, compositional, comparison)
        rules: Pre-computed rules (if None, will call predict_ans_rules)
        normalize: Whether to normalize the answer using Wikipedia
        model: Model for answer generation
        temperature: Temperature for generation
        
    Returns:
        Dict containing:
        - short_answer: Final short answer
        - long_answer: Explanation
        - rules: Rules used for generation
        - normalization: Normalization metadata
        - raw_answer: Answer before normalization
    """
    
    result: Dict[str, Any] = {
        "question": question,
        "context": context if isinstance(context, str) else " ".join(context),
        "key_entity": key_entity,
        "pipeline_steps": []
    }
    
    # Step 1: Predict answer rules (if not provided)
    if rules is None:
        rules = predict_answer_rules(
            question=question,
            dataset=dataset,
            type_q=type_q,
            model=model,
            temperature=temperature
        )
    
    result["rules"] = rules
    result["pipeline_steps"].append({
        "step": "predict_ans_rules",
        "output": {
            "question_type": rules.get("question_type"),
            "primary_rule": rules.get("primary_rule"),
            "needs_normalization": rules.get("needs_normalization", False)
        }
    })
    
    # Step 2: Generate answer with injected rules
    raw_answer = generate_answer_with_rules(
        question=question,
        context=context,
        rules=rules,
        key_entity=key_entity,
        dataset=dataset,
        type_q=type_q,
        model=model,
        temperature=temperature
    )
    
    result["raw_answer"] = raw_answer.copy()
    result["pipeline_steps"].append({
        "step": "gen_ans",
        "output": {
            "short_answer": raw_answer.get("short_answer"),
            "has_prefix_name_suffix": all(k in raw_answer for k in ["prefix", "name", "suffix"])
        }
    })
    
    # Step 3: Normalize answer using Wikipedia search
    # Only normalize if the rule indicates it's needed and normalize flag is True
    should_normalize = normalize and rules.get("needs_normalization", False)
    
    if should_normalize:
        normalized_answer = normalize_answer_with_wikipedia(
            raw_answer,
            should_normalize=True
        )
        result["short_answer"] = normalized_answer.get("short_answer", raw_answer.get("short_answer"))
        result["long_answer"] = normalized_answer.get("long_answer", raw_answer.get("long_answer"))
        result["normalization"] = normalized_answer.get("normalization", {})
        result["pipeline_steps"].append({
            "step": "norm_ans",
            "output": {
                "was_normalized": result["normalization"].get("was_normalized", False),
                "changes": result["normalization"].get("changes", [])
            }
        })
        
        # Preserve prefix/name/suffix if present
        for key in ["prefix", "name", "suffix"]:
            if key in normalized_answer:
                result[key] = normalized_answer[key]
            elif key in raw_answer:
                result[key] = raw_answer[key]
    else:
        result["short_answer"] = raw_answer.get("short_answer", "")
        result["long_answer"] = raw_answer.get("long_answer", "")
        skip_reason = "Normalization disabled" if not normalize else "Answer type does not require normalization"
        result["normalization"] = {
            "was_normalized": False,
            "changes": [],
            "skip_reason": skip_reason
        }
        
        for key in ["prefix", "name", "suffix"]:
            if key in raw_answer:
                result[key] = raw_answer[key]
    
    return result


def run_pipeline(
    question: str,
    contexts: List[Union[str, List[str]]],
    key_entity: Optional[str] = None,
    dataset: Optional[str] = None,
    type_q: Optional[str] = None,
    normalize: bool = True,
    gen_model: str = "gpt-4o-mini",
    select_model: str = "o3-mini",
    temperature: float = 0
) -> Dict[str, Any]:
    """
    Run the full answer generation pipeline on multiple contexts and select the best answer.
    
    Pipeline:
    For each context:
        1. predict_ans_rules → 2. gen_ans with injected rules → 3. norm_ans (Wikipedia search)
    Then:
        4. handle_multi_choice (LLM-based selection using o3-mini)
    
    Args:
        question: The question to answer
        contexts: List of contexts (each can be string or list of strings)
        key_entity: Optional key entity
        dataset: Dataset context
        type_q: Question type hint
        normalize: Whether to normalize answers
        gen_model: Model for answer generation (default: gpt-4o-mini)
        select_model: Model for answer selection (default: o3-mini)
        temperature: Temperature for generation
        
    Returns:
        Dict with structure matching ans.json:
        {
            "chosen_answer": {"short_answer": str, "long_answer": str},
            "reasoning": str,
            "answer_index": int,
            "answer_options": [
                {
                    "before_norm": {"short_answer": str, "long_answer": str},
                    "after_norm": {"short_answer": str, "long_answer": str},
                    "context_used": str,
                    "predicted_rules": dict,
                    "normalization_metadata": {"needs_normalization": bool, "was_normalized": bool},
                    "normalization_changes": list (optional)
                }
            ]
        }
    """
    
    if not contexts:
        return {
            "chosen_answer": {
                "short_answer": "Insufficient information",
                "long_answer": "No context provided."
            },
            "reasoning": "No contexts provided",
            "answer_index": 0,
            "answer_options": []
        }
    
    # Predict rules once (shared across all contexts)
    rules = predict_answer_rules(
        question=question,
        dataset=dataset,
        type_q=type_q,
        model=gen_model,
        temperature=temperature
    )
    
    # Process each context
    answer_options = []
    for idx, context in enumerate(contexts):
        answer_result = ans_for_single_context(
            question=question,
            context=context,
            key_entity=key_entity,
            dataset=dataset,
            type_q=type_q,
            rules=rules,
            normalize=normalize,
            model=gen_model,
            temperature=temperature
        )
        
        # Build answer option in ans.json format
        option = {
            "before_norm": {
                "short_answer": answer_result.get("raw_answer", {}).get("short_answer", ""),
                "long_answer": answer_result.get("raw_answer", {}).get("long_answer", "")
            },
            "after_norm": {
                "short_answer": answer_result.get("short_answer", ""),
                "long_answer": answer_result.get("long_answer", "")
            },
            "context_used": context if isinstance(context, str) else " ".join(context),
            "predicted_rules": rules,
            "normalization_metadata": {
                "needs_normalization": rules.get("needs_normalization", False),
                "was_normalized": answer_result.get("normalization", {}).get("was_normalized", False)
            }
        }
        
        # Add normalization changes if present
        normalization = answer_result.get("normalization", {})
        if normalization.get("was_normalized"):
            option["normalization_changes"] = normalization.get("changes", [])
        
        # Add prefix/name/suffix if present (for royal/noble answers)
        for key in ["prefix", "name", "suffix"]:
            if key in answer_result:
                option["after_norm"][key] = answer_result[key]
        
        answer_options.append(option)
    
    # Prepare candidates for selection (only after_norm versions)
    answer_candidates = [
        {
            "short_answer": opt["after_norm"]["short_answer"],
            "long_answer": opt["after_norm"]["long_answer"]
        }
        for opt in answer_options
    ]
    
    # Select best answer using LLM (o3-mini)
    selection_result = select_best_answer(
        question=question,
        answers=answer_candidates,
        model=select_model,
        temperature=temperature
    )
    
    # Build final output in ans.json format
    return {
        "chosen_answer": selection_result.get("chosen_answer", answer_candidates[0] if answer_candidates else {}),
        "reasoning": selection_result.get("reasoning", ""),
        "answer_index": selection_result.get("answer_index", 1),
        "answer_options": answer_options
    }


# Demo/Mock functions for testing without API calls
def mock_ans_for_single_context(
    question: str,
    context: str
) -> Dict[str, Any]:
    """
    Mock version for demonstration/testing without API calls.
    Simulates the pipeline output structure.
    """
    return {
        "question": question,
        "context": context[:100] + "..." if len(context) > 100 else context,
        "short_answer": "Mock answer",
        "long_answer": "This is a mock explanation.",
        "rules": {
            "question_type": "general_entity",
            "primary_rule": "general_entity",
            "secondary_rules": ["insufficient_info"]
        },
        "normalization": {
            "was_normalized": False,
            "changes": []
        },
        "pipeline_steps": [
            {"step": "predict_ans_rules", "output": {"primary_rule": "general_entity"}},
            {"step": "gen_ans", "output": {"short_answer": "Mock answer"}},
            {"step": "norm_ans", "output": {"was_normalized": False}}
        ]
    }


def demo_pipeline():
    """
    Demonstration of the pipeline with mock data.
    Shows the expected flow and output structure.
    """
    print("=" * 60)
    print("PIPELINE DEMONSTRATION (Mock Data)")
    print("=" * 60)
    
    # Mock inputs
    question = "Where was Albert Einstein born?"
    contexts = [
        "Albert Einstein was born on 14 March 1879 in Ulm, in the Kingdom of Württemberg in the German Empire.",
        "Einstein, the German-born physicist, spent his early childhood in Munich where his family moved shortly after his birth.",
        "The birthplace of Albert Einstein is Ulm, Germany. He later became a Swiss and American citizen."
    ]
    
    print(f"\nQuestion: {question}")
    print(f"\nNumber of contexts: {len(contexts)}")
    print("\nContexts:")
    for i, ctx in enumerate(contexts, 1):
        print(f"  [{i}] {ctx[:80]}...")
    
    print("\n" + "-" * 60)
    print("STEP 1: Process each context individually")
    print("-" * 60)
    
    # Simulate processing each context
    mock_answers = []
    for i, ctx in enumerate(contexts, 1):
        mock_result = mock_ans_for_single_context(question, ctx)
        mock_answers.append(mock_result)
        print(f"\nContext {i}:")
        print(f"  Pipeline steps: predict_ans_rules → gen_ans → norm_ans")
        print(f"  Short answer: {mock_result['short_answer']}")
        print(f"  Normalized: {mock_result['normalization']['was_normalized']}")
    
    print("\n" + "-" * 60)
    print("STEP 2: Select best answer using o3-mini")
    print("-" * 60)
    
    # Simulate selection
    print("\nCandidates submitted to o3-mini:")
    for i, ans in enumerate(mock_answers, 1):
        print(f"  [{i}] {ans['short_answer']}")
    
    print("\nSelection result (mock):")
    print("  Selected index: 1")
    print("  Reasoning: Context 1 provides the most direct and explicit answer")
    print("  Confidence: high")
    
    print("\n" + "-" * 60)
    print("FINAL OUTPUT STRUCTURE")
    print("-" * 60)
    
    final_output = {
        "final_answer": {
            "short_answer": "Ulm",
            "long_answer": "Albert Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire, on 14 March 1879."
        },
        "all_answers": [{"short_answer": f"Answer from context {i+1}"} for i in range(3)],
        "selection_result": {
            "answer_index": 1,
            "reasoning": "Context 1 provides the most direct answer",
            "confidence": "high"
        },
        "rules": {
            "question_type": "location",
            "primary_rule": "location"
        }
    }
    
    print(json.dumps(final_output, indent=2))
    print("\n" + "=" * 60)


# For testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_pipeline()
    else:
        print("Run with --demo for pipeline demonstration")
