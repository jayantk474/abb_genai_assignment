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
        Compliant pipeline:
        - Vector similarity search (FAISS) → candidate pool
        - Optional rerank
        - Return FINAL top-5 chunks
        """

        import numpy as np

        FINAL_K = self.cfg.top_k_retrieve  # must be 5 (assignment)
        CANDIDATE_K = 30  # internal candidate pool

        # ---- Get embedder safely ----
        embedder = (
                getattr(self, "embed_model", None)
                or getattr(self, "embedder", None)
                or getattr(self, "embedding_model", None)
        )
        if embedder is None:
            raise AttributeError("No embedding model found on RagSystem")

        # ---- Encode query ----
        qv = embedder.encode([q], normalize_embeddings=True)
        qv = np.array(qv).astype("float32")

        # ---- FAISS search (use real index inside wrapper) ----
        D, I = self.index.index.search(qv, CANDIDATE_K)

        candidates = []

        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue

            candidates.append({
                "text": self.index.texts[idx],
                "metadata": self.index.metadatas[idx],
                "score": float(score)
            })

        # ---- Optional reranking ----
        if self.reranker is not None and len(candidates) > 1:
            pairs = [(q, c["text"]) for c in candidates]
            rerank_scores = self.reranker.predict(pairs)

            for c, rs in zip(candidates, rerank_scores):
                c["_rerank_score"] = float(rs)

            candidates.sort(key=lambda x: x["_rerank_score"], reverse=True)
        else:
            # If no reranker, sort by similarity score
            candidates.sort(key=lambda x: x["score"], reverse=True)

        # ---- Final top-5 ----
        top = candidates[:FINAL_K]

        print("Retrieved:", len(top))

        return [
            {"text": c["text"], "metadata": c["metadata"]}
            for c in top
        ]

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
