from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, re
from typing import Optional

class HFCausalWrapper:
    def __init__(self, model_name, device=None, dtype="bfloat16", max_new_tokens=256, temperature=0.2, top_p=0.95):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, dtype) if dtype else None, device_map="auto")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tok.eos_token_id,
        )
        gen = self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return gen.strip()

    def classify_json(self, prompt: str) -> dict:
        # Expect a small JSON object {"label": "...", "confidence": 0.xx}
        txt = self.generate(prompt)
        # Try to extract JSON
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            # Fallback: try "Label: X, Confidence: 0.73"
            lm = re.search(r"[Ll]abel\s*[:\-]\s*([^\n,]+)", txt)
            cm = re.search(r"[Cc]onfidence\s*[:\-]\s*([0-9\.]+)", txt)
            return {"label": lm.group(1).strip() if lm else txt.strip()[:64], "confidence": float(cm.group(1)) if cm else None, "raw": txt}
        try:
            obj = json.loads(m.group(0))
            obj["raw"] = txt
            return obj
        except Exception:
            return {"label": txt.strip().split()[0], "confidence": None, "raw": txt}
