"""
MHBench – Reasoning Coherence Metrics
-------------------------------------
Evaluates how logically consistent, coherent, and contextually aligned
the model's reasoning or explanation is relative to the query and reference.

Metrics:
    • coherence_score        – semantic coherence between query and response
    • consistency_score      – semantic consistency between reference and response
    • reasoning_quality      – combined geometric mean of both
    • local_sentence_entropy – optional internal variation measure (fluency proxy)

Backend: Sentence-BERT (semantic similarity, cosine)

Author: Sepehr Ghamari
"""

from typing import List, Dict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import re


# -----------------------------------------------------------
# 1. Sentence splitter helper
# -----------------------------------------------------------

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def sent_split(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return sents if sents else [text]


# -----------------------------------------------------------
# 2. Coherence scoring core
# -----------------------------------------------------------

class ReasoningCoherenceScorer:
    """
    Measures reasoning coherence and logical consistency.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    @torch.inference_mode()
    def score(self, query: str, response: str, reference: str = None) -> Dict[str, float]:
        """
        Compute reasoning coherence and consistency.

        Args:
            query (str): Original user input or problem statement.
            response (str): Model-generated reasoning or explanation.
            reference (str, optional): Gold rationale or expected explanation.

        Returns:
            dict: {
                "coherence_score": float,
                "consistency_score": float | None,
                "reasoning_quality": float,
                "local_sentence_entropy": float
            }
        """
        if not response.strip():
            return {
                "coherence_score": None,
                "consistency_score": None,
                "reasoning_quality": None,
                "local_sentence_entropy": None
            }

        # Compute embeddings
        emb_query = self.model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        emb_resp = self.model.encode([response], convert_to_tensor=True, normalize_embeddings=True)

        coherence = util.cos_sim(emb_query, emb_resp).item()

        if reference:
            emb_ref = self.model.encode([reference], convert_to_tensor=True, normalize_embeddings=True)
            consistency = util.cos_sim(emb_ref, emb_resp).item()
        else:
            consistency = None

        # Combined reasoning quality (geometric mean)
        if consistency is not None:
            reasoning_quality = float(np.sqrt(max(coherence, 0) * max(consistency, 0)))
        else:
            reasoning_quality = coherence

        # Local coherence: variation across sentences
        sents = sent_split(response)
        if len(sents) > 1:
            emb_sents = self.model.encode(sents, convert_to_tensor=True, normalize_embeddings=True)
            cos_mat = util.cos_sim(emb_sents, emb_sents).cpu().numpy()
            upper_tri = cos_mat[np.triu_indices(len(sents), k=1)]
            local_ent = float(1 - np.mean(upper_tri))  # higher means more varied, less redundancy
        else:
            local_ent = 0.0

        return {
            "coherence_score": round(float(coherence), 4),
            "consistency_score": round(float(consistency), 4) if consistency is not None else None,
            "reasoning_quality": round(float(reasoning_quality), 4),
            "local_sentence_entropy": round(local_ent, 4)
        }


# -----------------------------------------------------------
# 3. MHBench compute() interface
# -----------------------------------------------------------

def compute(instance: Dict, response: Dict):
    """
    Unified MHBench compute() interface for reasoning coherence.

    Args:
        instance (dict): Unified benchmark instance containing 'prompt' and reference rationale.
        response (dict): Model-generated output (should include 'explanation' or 'response_text').

    Returns:
        dict: {
            "coherence_score": float,
            "consistency_score": float,
            "reasoning_quality": float,
            "local_sentence_entropy": float
        }
    """
    try:
        query = instance.get("input", {}).get("prompt", "")
        reference = (
            instance.get("reference", {}).get("summary") or
            instance.get("reference", {}).get("explanation") or
            instance.get("reference", {}).get("rationale") or
            ""
        )

        if isinstance(response, dict):
            resp_text = response.get("explanation") or response.get("response_text") or ""
        else:
            resp_text = str(response)

        scorer = ReasoningCoherenceScorer()
        result = scorer.score(query, resp_text, reference)
        return result

    except Exception as e:
        return {
            "coherence_score": None,
            "consistency_score": None,
            "reasoning_quality": None,
            "local_sentence_entropy": None,
            "error": str(e)
        }


# -----------------------------------------------------------
# 4. Manual test
# -----------------------------------------------------------

if __name__ == "__main__":
    instance = {
        "input": {"prompt": "Patient reports difficulty sleeping and constant worry about work."},
        "reference": {"summary": "The reasoning shows anxiety linked to chronic work stress."}
    }
    response = {"explanation": "The patient experiences persistent anxiety and insomnia caused by work-related stress."}

    print("=== MHBench Reasoning Coherence Test ===")
    print(compute(instance, response))
