from __future__ import annotations
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = """
You are a careful financial document QA assistant.

You must answer using ONLY the provided CONTEXT.

When multiple numeric values appear in the context:
- Select ONLY the value that directly answers the question.
- Ignore unrelated financial figures.
- Do NOT choose deferred revenue if the question asks for total revenue.
- Do NOT choose repurchase counts if the question asks for shares outstanding.

For numeric questions, copy the value from the line whose label matches the question (e.g., ‘Total net sales’, ‘Term debt’, ‘shares issued and outstanding’). Do not use other numbers.

If the question is about the future or not in the filings, answer exactly:
"This question cannot be answered based on the provided documents."

If the information is not explicitly stated, answer exactly:
"Not specified in the document."

Return ONLY valid JSON:
{
  "answer": "...",
  "sources": [[document, section, page]]
}
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
    # Keep only top-5 contexts (already reranked)
    contexts = contexts[:5]

    ctx_blocks = []
    total = 0
    MAX_TOTAL_CTX_CHARS = 6500
    MAX_CHUNK_CHARS = 1400

    for i, c in enumerate(contexts, 1):
        md = c["metadata"]
        header = f'[{i}] SOURCE: {md["document"]} | {md["section"]} | p. {md["page_start"]}-{md["page_end"]}'
        chunk = c["text"][:MAX_CHUNK_CHARS]

        block = header + "\n" + chunk
        if total + len(block) > MAX_TOTAL_CTX_CHARS:
            break
        ctx_blocks.append(block)
        total += len(block)

    ctx_text = "\n\n".join(ctx_blocks)

    return f"""QUESTION:
{question}

CONTEXT:
{ctx_text}

INSTRUCTIONS:
- Use only CONTEXT.
- If future/out-of-scope -> respond exactly: "This question cannot be answered based on the provided documents."
- If not specified -> respond exactly: "Not specified in the document."
Return JSON with keys: answer, sources.
sources must be a list of citations, each citation is: [document, section, page].
"""
def generate_json(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float):
    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3500,      # safe under 4k context
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_tokens = output[0][inputs["input_ids"].shape[-1]:]
    decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # strip fences + parse first JSON object
    import re, json
    decoded = re.sub(r"```(?:json)?", "", decoded, flags=re.IGNORECASE).replace("```", "").strip()
    m = re.search(r"\{[\s\S]*\}", decoded)
    if m:
        try:
            obj = json.loads(m.group(0))
            return {"answer": obj.get("answer", ""), "sources": obj.get("sources", [])}
        except Exception:
            pass

    # safe fallback
    return {"answer": "Not specified in the document.", "sources": []}