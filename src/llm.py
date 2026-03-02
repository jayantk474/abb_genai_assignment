from __future__ import annotations
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = """
You are a careful financial document QA assistant.
You must answer using ONLY the provided CONTEXT from SEC 10-K filings.

If the question is about the future, personal opinions, forecasts, or anything NOT contained in the filings, answer exactly:
"This question cannot be answered based on the provided documents."

If the question is in-scope but the documents do not specify the requested detail, answer exactly:
"Not specified in the document."

Always provide a short, direct answer ONLY.
Do NOT output JSON.
"""

def load_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Many instruct models expect a chat template; Phi-3 supports chat template in tokenizer.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,  # ↓ cuts RAM vs float32
        #low_cpu_mem_usage=True
    )
    model.eval()
    print("Model device:", next(model.parameters()).device)
    return tokenizer, model

def build_prompt(question: str, contexts: list[dict]) -> str:
    """
    Simple, robust prompt:
    - includes sources
    - ends with 'Answer:' so the model doesn't echo QUESTION/CONTEXT
    - keep context reasonably bounded to avoid GPU OOM
    """
    # Keep only first 5 contexts (already top-5)
    contexts = contexts[:5]

    # Light context caps to prevent OOM while keeping tables readable
    MAX_CHUNK_CHARS = 1600
    MAX_TOTAL_CTX_CHARS = 7000

    blocks = []
    total = 0
    for i, c in enumerate(contexts, 1):
        md = c["metadata"]
        header = f'[{i}] SOURCE: {md["document"]} | {md["section"]} | p. {md["page_start"]}-{md["page_end"]}'
        chunk = (c["text"] or "").strip()
        chunk = chunk[:MAX_CHUNK_CHARS]

        block = header + "\n" + chunk
        if total + len(block) > MAX_TOTAL_CTX_CHARS:
            break

        blocks.append(block)
        total += len(block)

    ctx_text = "\n\n".join(blocks)

    system = (
        "You are a careful financial document QA assistant.\n"
        "You must answer using ONLY the provided CONTEXT from SEC 10-K filings.\n"
        "If the question is about the future, personal opinions, forecasts, or anything NOT contained in the filings, answer exactly:\n"
        "\"This question cannot be answered based on the provided documents.\"\n"
        "If the question is in-scope but the documents do not specify the requested detail, answer exactly:\n"
        "\"Not specified in the document.\"\n"
        "Always provide a short, direct answer.\n"
    )

    return f"""{system}

CONTEXT:
{ctx_text}

QUESTION: {question}

Answer:"""


import torch
import re

@torch.inference_mode()
def generate_json(
    tokenizer,
    model,
    question: str,
    contexts: list[dict],
    max_new_tokens: int = 80,
    temperature: float = 0.0,
) -> str:
    prompt = build_prompt(question, contexts)

    # Ensure pad token exists (Phi-3 often uses eos as pad)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Deterministic generation
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only generated tokens (prevents prompt echo)
    gen_tokens = output[0][inputs["input_ids"].shape[-1]:]
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Clean common artifacts
    decoded = re.sub(r"```(?:json)?", "", decoded, flags=re.IGNORECASE).replace("```", "").strip()

    # Return only first meaningful line (prevents 'QUESTION:' echo + long rambles)
    lines = [ln.strip() for ln in decoded.splitlines() if ln.strip()]
    answer = lines[0] if lines else "Not specified in the document."

    return answer
