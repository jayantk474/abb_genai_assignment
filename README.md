# 10-K RAG Assignment (Apple 2024 + Tesla 2023)

This repo implements a **Retrieval-Augmented Generation (RAG)** system over two SEC 10‑K PDFs:
- Apple 2024 10‑K
- Tesla 2023 10‑K

It builds a FAISS vector index + BM25 hybrid retriever + cross-encoder re-ranker, then uses an open-access local LLM to answer questions **only using retrieved context**.

## Repo Structure

- `scripts/build_index.py` – parse PDFs, chunk, embed, and build FAISS index
- `scripts/run_questions.py` – answer the 13 questions and save JSON
- `src/` – chunking, retrieval, reranking, and LLM code
- `app.py` – required function: `answer_question(query: str) -> dict`
- `design_report.md` – 1-page design explanation

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## Build the index

You must have the two PDFs locally (paths below are examples):

```bash
python scripts/build_index.py \
  --apple_pdf data/apple_2024_10k.pdf \
  --tesla_pdf data/tesla_2023_10k.pdf \
  --out_dir artifacts/index
```

This will create:
- `artifacts/index/faiss.index`
- `artifacts/index/store.jsonl`

## Run the 13 evaluation questions

```bash
python scripts/run_questions.py --index_dir artifacts/index --out_json outputs/answers.json
```

Output format matches the assignment:

```json
[
  {"question_id": 1, "answer": "...", "sources": [["Apple 10-K","Item 8","p. 282"]]}
]
```

## Ask a custom question

```bash
python scripts/run_questions.py --single_question "What is Apple's total revenue for FY2024?"
```

## Required function interface

In `app.py`:

```python
def answer_question(query: str) -> dict:
    return {"answer": "...", "sources": [["Apple 10-K","Item 8","p. 282"]]}
```

## Notes for Colab / Kaggle
- The provided LLM is `microsoft/Phi-3-mini-4k-instruct`.
- For best results, enable a GPU runtime.
- If you hit memory limits, reduce `max_new_tokens` in `src/config.py` or switch to a smaller model.

## License
For academic use.
