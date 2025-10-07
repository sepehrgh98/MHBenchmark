"""
MHBench – Fluency Metrics
-------------------------
Computes intrinsic language-model fluency via:
    • Perplexity (PPL)  – for causal LMs (e.g., GPT-2, Falcon, LLaMA)
    • Pseudo-Perplexity (PLL) – for masked LMs (e.g., BERT, RoBERTa)

Author: Sepehr Ghamari
"""

import math
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)
from typing import List, Dict, Union


# -----------------------------------------------------------
# 1. Perplexity for causal language models
# -----------------------------------------------------------

class PerplexityScorer:
    """
    Perplexity evaluator for causal language models (e.g., GPT-2).
    """

    def __init__(self, ref_model: str = "gpt2", dtype: str = "bfloat16"):
        """
        Args:
            ref_model (str): Model name or path.
            dtype (str): torch dtype (e.g., 'bfloat16', 'float32').
        """
        try:
            torch_dtype = getattr(torch, dtype) if dtype else None
        except AttributeError:
            torch_dtype = torch.float32

        self.tok = AutoTokenizer.from_pretrained(ref_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            ref_model, torch_dtype=torch_dtype, device_map="auto"
        )

        # Ensure padding token exists
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        self.model.eval()

    @torch.inference_mode()
    def ppl(self, texts: List[str], batch_size: int = 4, max_length: int = 1024):
        """
        Compute perplexity for a list of texts.

        Args:
            texts (list[str]): Input texts.
            batch_size (int): Batch size.
            max_length (int): Max token length for truncation.

        Returns:
            dict: {
                "per_sample_ppl": [float],
                "mean_ppl": float
            }
        """
        all_ppl = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            ).to(self.model.device)
            out = self.model(**enc, labels=enc["input_ids"])
            losses = out.loss.detach().cpu()

            for loss in losses if losses.ndim > 0 else [losses]:
                all_ppl.append(math.exp(float(loss)))

        return {
            "per_sample_ppl": all_ppl,
            "mean_ppl": sum(all_ppl) / len(all_ppl) if all_ppl else None,
        }


# -----------------------------------------------------------
# 2. Pseudo-Perplexity for masked language models
# -----------------------------------------------------------

class PseudoPerplexityScorer:
    """
    Computes pseudo-perplexity (PLL) for masked language models (e.g., BERT).
    """

    def __init__(self, ref_model: str = "bert-base-uncased"):
        """
        Args:
            ref_model (str): Name or path of a masked LM (e.g., 'bert-base-uncased').
        """
        self.tok = AutoTokenizer.from_pretrained(ref_model)
        self.model = AutoModelForMaskedLM.from_pretrained(ref_model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def pll(self, text: str, max_length: int = 512):
        """
        Compute pseudo-perplexity for a single text sequence.

        Args:
            text (str): Input sentence or paragraph.
            max_length (int): Maximum length for tokenization.

        Returns:
            float: Pseudo-perplexity value.
        """
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(self.device)
        n_tokens = input_ids.size(1)
        total_loss = 0.0

        for i in range(n_tokens):
            masked = input_ids.clone()
            masked[0, i] = self.tok.mask_token_id
            logits = self.model(masked).logits
            log_prob = torch.log_softmax(logits[0, i], dim=-1)
            total_loss += -log_prob[input_ids[0, i]].item()

        return math.exp(total_loss / n_tokens)


# -----------------------------------------------------------
# 3. MHBench compute() interface
# -----------------------------------------------------------

def compute(instance: Dict, response: Union[Dict, str]):
    """
    MHBench compute() entrypoint for fluency metrics.

    Chooses between causal (PPL) and masked (PLL) LM depending on configuration.

    Args:
        instance (dict): Unified benchmark instance (optional metadata).
        response (dict | str): Model-generated response text.

    Returns:
        dict: {"mean_ppl": float, "mean_pll": float, "per_sample_ppl": list | None}
    """
    try:
        # Extract raw text
        if isinstance(response, dict):
            if "response_text" in response:
                text = response["response_text"]
            elif "explanation" in response:
                text = response["explanation"]
            else:
                text = str(response)
        else:
            text = str(response)

        if not text or len(text.strip()) == 0:
            return {"mean_ppl": None, "mean_pll": None}

        # Choose scoring strategy
        mode = instance.get("metadata", {}).get("fluency_mode", "causal")

        if mode == "causal":
            scorer = PerplexityScorer(ref_model="gpt2")
            res = scorer.ppl([text])
            return {"mean_ppl": res["mean_ppl"], "mean_pll": None}

        else:
            pll_scorer = PseudoPerplexityScorer("bert-base-uncased")
            pll_val = pll_scorer.pll(text)
            return {"mean_ppl": None, "mean_pll": pll_val}

    except Exception as e:
        return {"mean_ppl": None, "mean_pll": None, "error": str(e)}


# -----------------------------------------------------------
# 4. Manual test
# -----------------------------------------------------------

if __name__ == "__main__":
    instance = {"metadata": {"fluency_mode": "causal"}}
    response = {"response_text": "I understand how stressful work can be. Try to take breaks when you can."}
    print(compute(instance, response))
