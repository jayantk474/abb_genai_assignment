from __future__ import annotations
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = """You are a financial document extraction assistant.

You must extract the answer strictly from the provided CONTEXT.

Rules:
- Use ONLY the information in the CONTEXT.
- Do NOT use prior knowledge.
- Do NOT infer.
- Do NOT calculate.
- Do NOT summarize.
- Copy the answer exactly as written in the CONTEXT.

If the answer does not appear explicitly in the CONTEXT, respond exactly:
Not specified in the document.

If the question is about future forecasts, opinions, or anything not present in the CONTEXT, respond exactly:
This question cannot be answered based on the provided documents.

Return only the extracted answer text.
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
    contexts = contexts[:5]

    MAX_CHUNK_CHARS = 800
    MAX_TOTAL_CTX_CHARS = 3500

    blocks = []
    total = 0
    for i, c in enumerate(contexts, 1):
        md = c["metadata"]
        header = f'[{i}] SOURCE: {md["document"]} | {md["section"]} | p. {md["page_start"]}-{md["page_end"]}'
        chunk = (c["text"] or "").strip()[:MAX_CHUNK_CHARS]
        block = header + "\n" + chunk
        if total + len(block) > MAX_TOTAL_CTX_CHARS:
            break
        blocks.append(block)
        total += len(block)

    ctx_text = "\n\n".join(blocks)

    return f"""{SYSTEM_PROMPT}

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
