"""
MHBench Utility – LLM Interface
-------------------------------
Provides unified API wrappers for multiple LLM backends (OpenAI, Claude, Gemini, HuggingFace)
and normalizes their outputs to MHBench schema.

Output format (per response):
{
    "label": str | None,
    "confidence": float | None,
    "topk": list[str] | None,
    "explanation": str | None,
    "response_text": str | None
}

Author: Sepehr Ghamari
"""

import json
import time
from typing import Dict, Any
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# -----------------------------------------------------------
# 1. Setup model clients
# -----------------------------------------------------------

openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Configure Gemini
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception:
    pass


# -----------------------------------------------------------
# 2. Helper: safe JSON extraction
# -----------------------------------------------------------

def safe_json_extract(text: str) -> Dict[str, Any]:
    """
    Try to parse a JSON object from model output.
    """
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return {"response_text": text.strip()}
        return json.loads(text[start:end])
    except Exception:
        return {"response_text": text.strip()}


# -----------------------------------------------------------
# 3. LLM Call Functions
# -----------------------------------------------------------

def call_openai(prompt: str, model: str = "gpt-4o", max_tokens: int = 512, temperature: float = 0.2):
    for _ in range(3):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content
            return safe_json_extract(text)
        except Exception as e:
            time.sleep(2)
            last_err = str(e)
    return {"error": last_err}


def call_claude(prompt: str, model: str = "claude-3-5-sonnet", max_tokens: int = 512):
    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        return safe_json_extract(text)
    except Exception as e:
        return {"error": str(e)}


def call_gemini(prompt: str, model: str = "gemini-1.5-pro"):
    try:
        gen_model = genai.GenerativeModel(model)
        res = gen_model.generate_content(prompt)
        return safe_json_extract(res.text)
    except Exception as e:
        return {"error": str(e)}


def call_hf(model_path: str, prompt: str, max_tokens: int = 512):
    """
    For local HuggingFace models (e.g., LLaMA, Mixtral, Meditron)
    """
    try:
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens)
        text = tok.decode(out[0], skip_special_tokens=True)
        return safe_json_extract(text)
    except Exception as e:
        return {"error": str(e)}


# -----------------------------------------------------------
# 4. Dispatcher
# -----------------------------------------------------------

def query_llm(prompt: str, backend: str, **kwargs) -> Dict[str, Any]:
    """
    Dispatch query to target LLM backend.
    """
    backend = backend.lower()
    if backend.startswith("gpt"):
        return call_openai(prompt, model=backend)
    elif backend.startswith("claude"):
        return call_claude(prompt, model=backend)
    elif backend.startswith("gemini"):
        return call_gemini(prompt, model=backend)
    elif backend.startswith("mixtral") or backend.startswith("llama") or backend.startswith("meditron"):
        return call_hf(model_path=backend, prompt=prompt)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")
