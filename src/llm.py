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
        ctx_blocks.append(header + "\n" + c["text"][:2500])
    ctx_text = "\n\n".join(ctx_blocks)


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

        # Get tokenized inputs properly
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        )

        inputs = {"input_ids": input_ids}

    else:
        text = SYSTEM_PROMPT + "\n\n" + prompt
        inputs = tokenizer(text, return_tensors="pt")

    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    #  decode ONLY new tokens
    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = output[0][input_length:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Extract JSON
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

    return {
        "answer": decoded,
        "sources": []
    }