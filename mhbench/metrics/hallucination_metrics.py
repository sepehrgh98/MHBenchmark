"""
MHBench – Hallucination / Factual Consistency Metrics
-----------------------------------------------------
Scores how well a model's statements are supported by a reference source
(e.g., DSM-5 text, gold rationale, or dialogue context).

Metrics:
    • mean_factuality: average entailment probability (↑ better)
    • hallucination_rate: % of samples below entailment threshold (↓ better)
    • per-sample entailment scores

Backend: RoBERTa-large-MNLI (Natural Language Inference)

Author: Sepehr Ghamari
"""

from typing import List, Dict
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -----------------------------------------------------------
# 1. Sentence splitting helper
# -----------------------------------------------------------

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def simple_sent_split(text: str) -> List[str]:
    """Light sentence splitter using punctuation."""
    text = (text or "").strip()
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return sents if sents else [text]


# -----------------------------------------------------------
# 2. NLI-based factuality scorer
# -----------------------------------------------------------

class NLIFactualityScorer:
    """
    Uses an entailment model (e.g., roberta-large-mnli)
    to score factual support for each response.
    """

    def __init__(self, model_name: str = "roberta-large-mnli"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.entail_idx = 2  # label order: [contradiction, neutral, entailment]

    @torch.inference_mode()
    def pair_scores(self, premises: List[str], hypotheses: List[str], batch_size: int = 8) -> List[float]:
        """Return P(entailment) for each (premise, hypothesis) pair."""
        probs = []
        for i in range(0, len(premises), batch_size):
            p_batch = premises[i:i + batch_size]
            h_batch = hypotheses[i:i + batch_size]
            enc = self.tok(
                p_batch, h_batch,
                return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            logits = self.model(**enc).logits
            p_ent = torch.softmax(logits, dim=-1)[:, self.entail_idx]
            probs.extend(p_ent.detach().cpu().tolist())
        return probs

    def score(
        self,
        sources: List[str],
        responses: List[str],
        agg: str = "mean",
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute factuality and hallucination metrics.

        Args:
            sources: reference texts (e.g., DSM snippet)
            responses: generated answers/explanations
            agg: 'mean' or 'min' aggregation across sentences
            threshold: entailment probability threshold for hallucination detection
        """
        assert len(sources) == len(responses), "sources and responses must match in length"

        per_sample = []
        premises, hypotheses, idx_map = [], [], []

        for src, resp in zip(sources, responses):
            sents = simple_sent_split(resp)
            start = len(hypotheses)
            premises.extend([src] * len(sents))
            hypotheses.extend(sents)
            idx_map.append(list(range(start, start + len(sents))))

        flat_scores = self.pair_scores(premises, hypotheses)

        for idxs in idx_map:
            sent_scores = [flat_scores[i] for i in idxs]
            val = np.mean(sent_scores) if agg == "mean" else np.min(sent_scores)
            per_sample.append(val)

        mean_fact = float(np.mean(per_sample))
        hall_rate = float(np.mean(np.array(per_sample) < threshold))

        return {
            "mean_factuality": mean_fact,
            "hallucination_rate": hall_rate,
            "sample_entailment": [float(x) for x in per_sample],
        }


# -----------------------------------------------------------
# 3. MHBench compute() interface
# -----------------------------------------------------------

def compute(instance: Dict, response: Dict):
    """
    MHBench compute() entrypoint for factual consistency metrics.

    Args:
        instance (dict): Unified benchmark instance, with reference text or rationale.
        response (dict): Model-generated output.

    Returns:
        dict: {
            "mean_factuality": float,
            "hallucination_rate": float,
            "sample_entailment": list[float]
        }
    """
    try:
        # Extract relevant fields
        reference = instance.get("reference", {}).get("explanation") \
                    or instance.get("reference", {}).get("source_text") \
                    or instance.get("reference", {}).get("label") \
                    or instance.get("reference", {}).get("response_text")

        candidate = response.get("explanation") \
                    or response.get("response_text") \
                    or str(response)

        if not reference or not candidate:
            return {
                "mean_factuality": None,
                "hallucination_rate": None,
                "sample_entailment": []
            }

        scorer = NLIFactualityScorer()
        results = scorer.score(
            [reference], [candidate],
            agg=instance.get("metadata", {}).get("agg_mode", "mean"),
            threshold=instance.get("metadata", {}).get("fact_threshold", 0.5)
        )
        return results

    except Exception as e:
        return {
            "mean_factuality": None,
            "hallucination_rate": None,
            "sample_entailment": [],
            "error": str(e)
        }


# -----------------------------------------------------------
# 4. Manual test
# -----------------------------------------------------------

if __name__ == "__main__":
    instance = {
        "reference": {
            "explanation": "Generalized Anxiety Disorder involves excessive worry for at least 6 months and symptoms such as restlessness and sleep disturbance."
        }
    }
    response = {
        "explanation": "The patient shows months of worrying and sleep problems, consistent with GAD."
    }

    print("=== MHBench Hallucination Metric Test ===")
    print(compute(instance, response))
