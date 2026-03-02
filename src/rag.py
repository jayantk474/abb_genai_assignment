from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from .config import RagConfig
from .vector_store import VectorIndex, search as faiss_search
from .retrieval import HybridRetriever, hybrid_rank
from .llm import load_llm, build_prompt, generate_json

class RagSystem:
    def __init__(self, cfg: RagConfig, index: VectorIndex):
        self.cfg = cfg
        self.index = index
        self.embedder = SentenceTransformer(cfg.embed_model_name)
        self.bm25 = HybridRetriever(index.texts)
        self.reranker = CrossEncoder(cfg.reranker_model_name) if cfg.reranker_model_name else None
        self.tokenizer, self.model = load_llm(cfg.llm_model_name)

    def embed_query(self, q: str) -> np.ndarray:
        vec = self.embedder.encode([q], normalize_embeddings=True)
        return vec[0]

    def retrieve(self, q: str):
        """
        Simple pipeline:
        1) Vector similarity search (FAISS) -> top-5
        2) Optional rerank (CrossEncoder) of those same 5
        Returns: list of dicts: {"text":..., "metadata":...}
        """
        # 1) Vector similarity top-5
        qv = self.embedder.encode([q], normalize_embeddings=True)[0]
        vec_hits = faiss_search(self.index, qv, top_k=self.cfg.top_k_retrieve)  # should be 5


        candidates = vec_hits

        # 2) Rerank
        if self.reranker is not None and self.cfg.top_k_rerank > 0 and len(candidates) > 1:
            pairs = [(q, c["text"]) for c in candidates]
            scores = self.reranker.predict(pairs)  # higher is better

            for c, s in zip(candidates, scores):
                c["_rerank_score"] = float(s)

            candidates = sorted(candidates, key=lambda x: x.get("_rerank_score", -1e9), reverse=True)

        # Final: return top-5 (same count as required)
        top = candidates[: self.cfg.top_k_retrieve]
        return [{"text": c["text"], "metadata": c["metadata"]} for c in top]

    def answer_question(self, q: str):
        contexts = self.retrieve(q)

        # Build sources from retrieved chunks (always from top-5)
        sources = []
        for c in contexts:
            md = c["metadata"]
            # exact requested format: ["Apple 10-K", "Item 8", "p. 28"]
            sources.append([md["document"], md["section"], f"p. {md['page_start']}"])

        # LLM answers only (no JSON from model)
        answer_text = generate_json(
            tokenizer=self.llm_tokenizer,
            model=self.llm_model,
            question=q,
            contexts=contexts,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
        )

        # Enforce rubric rules on sources
        if answer_text in [
            "This question cannot be answered based on the provided documents.",
            "Not specified in the document.",
        ]:
            sources = []

        return {"answer": answer_text, "sources": sources}
