from __future__ import annotations
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = """You are a careful financial document QA assistant.
You must answer using ONLY the provided CONTEXT from SEC 10-K filings.
If the question is about the future, personal opinions, forecasts, or anything NOT contained in the filings, answer exactly:
"This question cannot be answered based on the provided documents."
If the question is in-scope but the documents do not specify the requested detail, answer exactly:
"Not specified in the document."
Always provide a short, direct answer.
Also return supporting sources as a list of citations formatted like:
["Apple 10-K", "Item 8", "p. 282"]
Only cite sources that appear in the provided context metadata.
"""

def load_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Many instruct models expect a chat template; Phi-3 supports chat template in tokenizer.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return tokenizer, model

def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    # contexts: list of {"text": ..., "metadata": {...}}
    ctx_blocks = []
    for i, c in enumerate(contexts, start=1):
        m = c["metadata"]
        doc = m.get("document","")
        sec = m.get("section","Unknown section")
        pstart = m.get("page_start","?")
        pend = m.get("page_end","?")
        header = f"[{i}] SOURCE: {doc} | {sec} | p. {pstart}" + (f"-{pend}" if pend != pstart else "")
        ctx_blocks.append(header + "\n" + c["text"])
    ctx_text = "\n\n".join(ctx_blocks)
    MAX_CONTEXT_CHARS = 3500
    ctx_text = ctx_text[:MAX_CONTEXT_CHARS]

    user = f"""QUESTION:
{question}

CONTEXT:
{ctx_text}

INSTRUCTIONS:
- Use only CONTEXT.
- If future/out-of-scope -> respond with the exact out-of-scope sentence.
- If not specified -> respond with the exact not-specified sentence.
- Otherwise answer concisely.
Return JSON with keys: answer, sources.
sources must be a list of citations, each citation is: [document, section, page].
"""
    # If chat template exists, we will apply later.
    return user

def generate_json(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float):

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        )
    else:
        text = SYSTEM_PROMPT + "\n\n" + prompt
        inputs = tokenizer(text, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = inputs.to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )


    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Extract JSON safely
    import re, json
    match = re.search(r"\{[\s\S]*\}", decoded)

    if match:
        try:
            obj = json.loads(match.group(0))
            return {
                "answer": obj.get("answer", "").strip(),
                "sources": obj.get("sources", [])
            }
        except:
            pass

    # fallback
    return {
        "answer": decoded,
        "sources": []
    }
