# Design Report (1 page)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) QA system over **Apple 2024 10‑K** and **Tesla 2023 10‑K** PDFs. The system answers questions using **only** retrieved filing excerpts and returns answers with citations.

## Chunking Strategy
- We extract text **page-by-page** from PDFs.
- We chunk by **character length** for reliability without tokenizer dependencies:
  - `chunk_chars=3500`, `chunk_overlap_chars=400`
- Each chunk stores metadata:
  - `document` (Apple 10‑K / Tesla 10‑K)
  - `page_start`, `page_end`
  - `section` (best-effort: detects “Item X”, “Note N”, “Signature page”)

## Retrieval
1. **Vector retrieval**: FAISS cosine similarity (inner product on normalized embeddings) using `all-MiniLM-L6-v2`.
2. **Hybrid scoring**: We combine vector similarity with **BM25** (rank-bm25). This improves recall for numeric / exact-match queries common in 10‑Ks.
3. **Top‑k policy**: retrieve top-20 candidates, then select final top‑5.

## Re-ranker Justification
We re-rank candidates using the cross-encoder `ms-marco-MiniLM-L-6-v2`.
- Embedding similarity is fast but sometimes ranks semantically related chunks above the *exact* table/note containing the answer.
- Cross-encoders score the query–chunk pair jointly and typically improve precision on finance QA (e.g., debt totals, shares outstanding).
- This reduces hallucinations by ensuring the final context is tightly relevant.

## LLM Choice
We use an open-access local model: `microsoft/Phi-3-mini-4k-instruct` (via HuggingFace Transformers).
- Small enough to run in Colab; works best with GPU.
- Inference prompt enforces “**use only context**” and returns structured JSON.

## Handling Out-of-Scope & Missing Information
The system uses strict rules:
- If the question asks for forecasts/future/attributes not in filings →  
  **"This question cannot be answered based on the provided documents."**
- If the question is in-scope but the detail is absent →  
  **"Not specified in the document."**
- For non-refusal answers, we attach citations derived from retrieved chunk metadata only.

## Reproducibility
- `scripts/build_index.py` builds the FAISS index from the two PDFs.
- `scripts/run_questions.py` runs the 13 fixed questions and writes `outputs/answers.json`.
- `app.py` exposes `answer_question(query: str) -> dict`.
