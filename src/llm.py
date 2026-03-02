from __future__ import annotations
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = """You are a financial document extraction assistant.

Use ONLY the information in the provided CONTEXT.

Rules:
- Use only the CONTEXT.
- Do NOT use prior knowledge.
- If the question requires a percentage or ratio, you may compute it ONLY using numbers explicitly present in the CONTEXT.
- If the answer does not appear or cannot be derived directly from the CONTEXT, respond exactly:
Not specified in the document.
- If the question is about future forecasts, opinions, or anything not present in the CONTEXT, respond exactly:
This question cannot be answered based on the provided documents.

Return only the final answer.
Do not explain.
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
    print("Retrieved:", len(contexts))
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
def generate_json(tokenizer, model, *args, **kwargs) -> str:
    """Generate the completion for our RAG prompt.

    This function is intentionally tolerant to minor signature mismatches
    to prevent integration breakages across notebooks/scripts.

    Supported call patterns:
    - generate_json(tokenizer, model, question, contexts, max_new_tokens=..., temperature=...)
    - generate_json(tokenizer, model, prompt, max_new_tokens=..., temperature=...)
    - generate_json(..., question=..., contexts=...)
    - generate_json(..., prompt=...)

    Always returns ONLY the generated answer text (no prompt echo).
    """

    question: Optional[str] = kwargs.pop("question", None)
    contexts = kwargs.pop("contexts", None)
    prompt: Optional[str] = kwargs.pop("prompt", None)
    max_new_tokens: int = int(kwargs.pop("max_new_tokens", 80))
    temperature: float = float(kwargs.pop("temperature", 0.0))

    # Positional parsing
    # (question, contexts, [max_new_tokens], [temperature])
    # or (prompt, [max_new_tokens], [temperature])
    if prompt is None and question is None and contexts is None:
        if len(args) >= 2:
            question = args[0]
            contexts = args[1]
            if len(args) >= 3:
                max_new_tokens = int(args[2])
            if len(args) >= 4:
                temperature = float(args[3])
        elif len(args) >= 1:
            prompt = args[0]
            if len(args) >= 2:
                max_new_tokens = int(args[1])
            if len(args) >= 3:
                temperature = float(args[2])

    if prompt is None:
        if question is None or contexts is None:
            raise TypeError(
                "generate_json expected either (question, contexts) or (prompt). "
                f"Got args={len(args)} and missing question/contexts/prompt."
            )
        prompt = build_prompt(str(question), contexts)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_tokens = output[0][input_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()