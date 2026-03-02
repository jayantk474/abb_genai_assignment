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
        1) FAISS similarity search -> top_k_retrieve (5)
        2) Rerank those 5 (if reranker enabled)
        Returns list of {"text":..., "metadata":...}
        """
        # Get embedder attribute safely (matches your repo)
        embedder = (
                getattr(self, "embed_model", None)
                or getattr(self, "embedder", None)
                or getattr(self, "embedding_model", None)
        )
        if embedder is None:
            raise AttributeError("No embedding model found on RagSystem")

        qv = embedder.encode([q], normalize_embeddings=True)[0]

        # FAISS top-5
        raw_hits = faiss_search(self.index, qv, top_k=self.cfg.top_k_retrieve)

        # ---- Normalize hits to dicts: {"text":..., "metadata":..., "score":...} ----
        candidates = []
        for h in raw_hits:
            if isinstance(h, dict):
                # already normalized
                candidates.append(h)
                continue

            # tuple/list formats:
            if isinstance(h, (tuple, list)):
                # Common cases:
                # (score, doc)  where doc is dict-like
                if len(h) == 2 and isinstance(h[1], dict):
                    score, doc = h
                    candidates.append(
                        {"text": doc.get("text", ""), "metadata": doc.get("metadata", {}), "score": float(score)}
                    )
                    continue

                # (doc, score)
                if len(h) == 2 and isinstance(h[0], dict):
                    doc, score = h
                    candidates.append(
                        {"text": doc.get("text", ""), "metadata": doc.get("metadata", {}), "score": float(score)}
                    )
                    continue

                # (idx, score) or (score, idx) -> use idx to fetch stored chunk
                if len(h) == 2 and all(isinstance(x, (int, float)) for x in h):
                    a, b = h
                    # try interpret first as idx
                    idx = int(a)
                    score = float(b)
                    if hasattr(self, "store") and hasattr(self.store, "get"):
                        doc = self.store.get(idx)
                        candidates.append({"text": doc["text"], "metadata": doc["metadata"], "score": score})
                        continue
                    # otherwise can't resolve -> skip
                    continue

            # unknown format -> skip
            continue

        # If reranker enabled, rerank these 5
        if self.reranker is not None and len(candidates) > 1:
            pairs = [(q, c["text"]) for c in candidates]
            scores = self.reranker.predict(pairs)
            for c, s in zip(candidates, scores):
                c["_rerank_score"] = float(s)
            candidates.sort(key=lambda x: x["_rerank_score"], reverse=True)

        # final top-5
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
            tokenizer=self.tokenizer,
            model=self.model,
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
