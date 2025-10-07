"""
MHBench – Diversity & Expressivity Metrics
------------------------------------------
Measures linguistic variety, expressivity, and repetitiveness
of model-generated text responses.

Metrics:
    • type_token_ratio      – lexical diversity
    • unique_word_ratio     – fraction of distinct words
    • mean_length, std_length – response length stats
    • avg_repeat_ratio      – average self-repetition within responses
    • diversity_score       – combined weighted metric

Author: Sepehr Ghamari
"""

from typing import List, Dict
from collections import Counter
import numpy as np
import re


# -----------------------------------------------------------
# 1. Tokenization utility
# -----------------------------------------------------------

def simple_tokenize(text: str) -> List[str]:
    """Lightweight word tokenizer using regex."""
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


# -----------------------------------------------------------
# 2. Core diversity computation
# -----------------------------------------------------------

def compute_diversity_metrics(responses: List[str]) -> Dict[str, float]:
    """
    Compute lexical and repetition-based diversity statistics.

    Args:
        responses: list of model-generated strings

    Returns:
        dict with multiple diversity metrics
    """
    if not responses:
        return {
            "type_token_ratio": None,
            "unique_word_ratio": None,
            "mean_length": None,
            "std_length": None,
            "avg_repeat_ratio": None,
            "diversity_score": None
        }

    all_tokens = []
    lengths = []
    repeat_ratios = []

    for resp in responses:
        tokens = simple_tokenize(resp)
        if not tokens:
            continue
        lengths.append(len(tokens))
        all_tokens.extend(tokens)

        # Repetition ratio within same response
        freq = Counter(tokens)
        repeats = sum(v - 1 for v in freq.values() if v > 1)
        repeat_ratios.append(repeats / len(tokens))

    vocab_size = len(set(all_tokens))
    total_tokens = len(all_tokens)

    type_token_ratio = vocab_size / total_tokens if total_tokens > 0 else 0.0
    unique_word_ratio = np.mean(
        [len(set(simple_tokenize(r))) / max(1, len(simple_tokenize(r))) for r in responses]
    )

    mean_len = float(np.mean(lengths)) if lengths else 0.0
    std_len = float(np.std(lengths)) if lengths else 0.0
    avg_repeat = float(np.mean(repeat_ratios)) if repeat_ratios else 0.0

    # Composite diversity score (higher = better)
    diversity_score = float(type_token_ratio * (1 - avg_repeat))

    return {
        "type_token_ratio": round(type_token_ratio, 4),
        "unique_word_ratio": round(unique_word_ratio, 4),
        "mean_length": round(mean_len, 2),
        "std_length": round(std_len, 2),
        "avg_repeat_ratio": round(avg_repeat, 4),
        "diversity_score": round(diversity_score, 4),
    }


# -----------------------------------------------------------
# 3. MHBench compute() interface
# -----------------------------------------------------------

def compute(instance, response):
    """
    MHBenchmark-compatible compute() function.

    Args:
        instance (dict): Unified benchmark instance.
        response (dict): Model's generated response (text or structured).

    Returns:
        dict: Diversity & expressivity metrics.
    """
    try:
        # Extract response text — support multiple forms
        if isinstance(response, dict):
            # Prefer a known response field if present
            if "response_text" in response:
                text = response["response_text"]
            elif "output" in response:
                text = response["output"]
            else:
                text = str(response)
        else:
            text = str(response)

        # Some tasks (like U7) may produce multiple candidate outputs
        responses = [text] if isinstance(text, str) else text
        return compute_diversity_metrics(responses)

    except Exception as e:
        return {
            "type_token_ratio": None,
            "unique_word_ratio": None,
            "mean_length": None,
            "std_length": None,
            "avg_repeat_ratio": None,
            "diversity_score": None,
            "error": str(e),
        }


# -----------------------------------------------------------
# 4. Manual test
# -----------------------------------------------------------

if __name__ == "__main__":
    response = {
        "response_text": "I understand how difficult it can be to manage stress at work. "
                         "Remember to take care of yourself and focus on small steps."
    }
    instance = {"task_id": "U7"}
    print(compute(instance, response))
