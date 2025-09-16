import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class PerplexityScorer:
    def __init__(self, ref_model="gpt2", dtype="bfloat16"):
        self.tok = AutoTokenizer.from_pretrained(ref_model)
        self.model = AutoModelForCausalLM.from_pretrained(ref_model, torch_dtype=getattr(torch, dtype) if dtype else None, device_map="auto")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    @torch.inference_mode()
    def ppl(self, texts):
        import math
        ppls=[]
        for t in texts:
            enc = self.tok(t, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
            out = self.model(**enc, labels=enc["input_ids"])
            # mean per-token loss on target
            loss = out.loss.item()
            ppls.append(math.exp(loss))
        return sum(ppls)/len(ppls) if ppls else None
