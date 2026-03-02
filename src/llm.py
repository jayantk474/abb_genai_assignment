from __future__ import annotations

from typing import Optional
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = """You are a strict financial document QA system.

Use ONLY the information in the provided CONTEXT.

Rules (must follow exactly):
- Use only the CONTEXT. Do not use prior knowledge.
- Do NOT output citations or headers like "SOURCE:" or "Document:".
- Output must be ONLY the answer text, on ONE line.
- If the answer does not appear or cannot be derived directly from the CONTEXT, respond EXACTLY:
Not specified in the document.
- If the question is out-of-scope for the provided documents (e.g., stock price forecasts, 2025 Apple CFO, Tesla HQ color), respond EXACTLY:
This question cannot be answered based on the provided documents.
- If a percentage/ratio is requested, compute it ONLY using numbers explicitly present in the CONTEXT.

Return only the final answer (one line)."""


def load_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return tokenizer, model


def build_prompt(question: str, contexts: list[dict]) -> str:
    # Keep more context: table rows often need more than 800 chars.
    contexts = contexts[:8]

    MAX_CHUNK_CHARS = 2200
    MAX_TOTAL_CTX_CHARS = 14000

    blocks = []
    total = 0
    for i, c in enumerate(contexts, 1):
        md = c.get("metadata", {})
        header = f'[{i}] SOURCE: {md.get("document","?")} | {md.get("section","?")} | p. {md.get("page_start","?")}-{md.get("page_end","?")}'
        chunk = (c.get("text") or "").strip()[:MAX_CHUNK_CHARS]
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


@torch.inference_mode()
def generate_json(tokenizer, model, *args, **kwargs) -> str:
    """Generate the completion for our RAG prompt.

    Returns ONLY the generated answer text (one line).
    """

    question: Optional[str] = kwargs.pop("question", None)
    contexts = kwargs.pop("contexts", None)
    prompt: Optional[str] = kwargs.pop("prompt", None)
    max_new_tokens: int = int(kwargs.pop("max_new_tokens", 60))
    temperature: float = float(kwargs.pop("temperature", 0.0))

    # Positional parsing
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
            raise TypeError("generate_json expected either (question, contexts) or (prompt).")
        prompt = build_prompt(str(question), contexts)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 1e-6

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 0.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Hard stop: one line only
    text = text.split("\n")[0].strip()

    # If the model starts inventing a new "QUESTION:", treat it as failure -> not specified.
    if re.match(r"^QUESTION\s*:\s*", text, re.IGNORECASE):
        return "Not specified in the document."

    # Normalize exact required strings
    if text.lower().startswith("not specified"):
        return "Not specified in the document."
    if text.lower().startswith("this question cannot be answered"):
        return "This question cannot be answered based on the provided documents."

    # Avoid returning headers
    if text.lower().startswith("document:") or text.lower().startswith("source:"):
        return "Not specified in the document."

    return text
