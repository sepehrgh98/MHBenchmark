"""
MHBench – Explanation Metrics
-----------------------------
Evaluates the quality of model explanations and rationales using:
    • BERTScore – semantic faithfulness & similarity
    • BLEU / ROUGE-L – lexical overlap (optional)
    • Human trust & consistency scaffolds – for subjective scoring

Author: Sepehr Ghamari
"""

import numpy as np
from typing import List, Dict
from evaluate import load


# -----------------------------------------------------------
# 1. Semantic Similarity (BERTScore)
# -----------------------------------------------------------

class BERTScoreScorer:
    """
    Computes BERTScore as a semantic similarity measure between
    model explanations (candidates) and gold rationales (references).
    """

    def __init__(self, model_type: str = "bert-base-uncased"):
        """
        Args:
            model_type (str): Model name for BERTScore backbone.
                              Common choices: 'bert-base-uncased', 'roberta-large'.
        """
        self.metric = load("bertscore")
        self.model_type = model_type

    def score(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """
        Compute BERTScore (precision, recall, F1).

        Args:
            references (list[str]): Gold rationales.
            candidates (list[str]): Model-generated rationales.

        Returns:
            dict: {
                "mean_bertscore": float,
                "precision": float,
                "recall": float,
                "f1": float
            }
        """
        if len(references) != len(candidates):
            raise ValueError("references and candidates must have the same length")

        results = self.metric.compute(
            predictions=candidates,
            references=references,
            model_type=self.model_type,
            lang="en"
        )

        return {
            "mean_bertscore": float(np.mean(results["f1"])),
            "precision": float(np.mean(results["precision"])),
            "recall": float(np.mean(results["recall"])),
            "scores_f1": [float(s) for s in results["f1"]],
        }


# -----------------------------------------------------------
# 2. BLEU & ROUGE-L (lexical overlap)
# -----------------------------------------------------------

class LexicalMetrics:
    """
    Computes BLEU and ROUGE-L scores to measure lexical overlap.
    """

    def __init__(self):
        self.bleu = load("bleu")
        self.rouge = load("rouge")

    def score(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """
        Args:
            references (list[str]): Gold rationales.
            candidates (list[str]): Model rationales.

        Returns:
            dict: {"bleu": float, "rougeL": float}
        """
        bleu_score = self.bleu.compute(predictions=candidates, references=references)["bleu"]
        rouge_score = self.rouge.compute(predictions=candidates, references=references)["rougeL"]
        return {"bleu": bleu_score, "rougeL": rouge_score}


# -----------------------------------------------------------
# 3. Human evaluation scaffolding
# -----------------------------------------------------------

def compute_human_explanation_scores(human_ratings: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate human-rated explanation scores (trustworthiness, consistency, clarity).

    Args:
        human_ratings (list[dict]):
            Each dict must contain:
                {"trust": float, "consistency": float, "clarity": float}

    Returns:
        dict: {"trust_mean": float, "consistency_mean": float, "clarity_mean": float}
    """
    trust = [r.get("trust", np.nan) for r in human_ratings]
    cons = [r.get("consistency", np.nan) for r in human_ratings]
    clar = [r.get("clarity", np.nan) for r in human_ratings]

    return {
        "trust_mean": float(np.nanmean(trust)),
        "consistency_mean": float(np.nanmean(cons)),
        "clarity_mean": float(np.nanmean(clar)),
    }


# -----------------------------------------------------------
# 4. MHBench Unified compute() interface
# -----------------------------------------------------------

def compute(instance, response):
    """
    MHBench compute() function for explanation and rationale evaluation.

    Supports:
      • BERTScore (semantic similarity)
      • BLEU & ROUGE-L (lexical overlap)
      • Human / LLM trust scaffolds if available

    Args:
        instance (dict): Unified task instance.
        response (dict): Model's generated response.

    Returns:
        dict: All computed metrics (semantic + lexical + optional human)
    """
    try:
        reference_expl = instance.get("reference", {}).get("explanation", "")
        candidate_expl = (
            response.get("explanation")
            if isinstance(response, dict)
            else str(response)
        )

        if not reference_expl or not candidate_expl:
            return {
                "mean_bertscore": None,
                "precision": None,
                "recall": None,
                "bleu": None,
                "rougeL": None,
                "trust_mean": None,
                "consistency_mean": None,
                "clarity_mean": None,
            }

        # --- Compute semantic similarity (BERTScore)
        bert = BERTScoreScorer(model_type="bert-base-uncased")
        bert_res = bert.score([reference_expl], [candidate_expl])

        # --- Compute lexical overlap (BLEU + ROUGE-L)
        lex = LexicalMetrics()
        lex_res = lex.score([reference_expl], [candidate_expl])

        # --- Optionally, if human/LLM trust ratings provided
        human_scores = response.get("ratings") if isinstance(response, dict) else None
        if human_scores:
            human_res = compute_human_explanation_scores([human_scores])
        else:
            human_res = {"trust_mean": None, "consistency_mean": None, "clarity_mean": None}

        return {**bert_res, **lex_res, **human_res}

    except Exception as e:
        return {
            "mean_bertscore": None,
            "precision": None,
            "recall": None,
            "bleu": None,
            "rougeL": None,
            "trust_mean": None,
            "consistency_mean": None,
            "clarity_mean": None,
            "error": str(e),
        }


# -----------------------------------------------------------
# 5. Manual test
# -----------------------------------------------------------

if __name__ == "__main__":
    instance = {
        "reference": {
            "explanation": "The patient expresses anxiety due to ongoing stress at work."
        }
    }
    response = {
        "explanation": "The model identifies stress at work as the main cause of anxiety.",
        "ratings": {"trust": 4.5, "consistency": 4.2, "clarity": 4.0},
    }

    print(compute(instance, response))
