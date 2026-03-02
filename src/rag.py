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
        # Convenience attribute used by some scripts/notebooks
        self.device = getattr(self.model, "device", None)

    def embed_query(self, q: str) -> np.ndarray:
        vec = self.embedder.encode([q], normalize_embeddings=True)
        return vec[0]

    def retrieve(self, q: str) -> list[dict]:
        """Return the FINAL top-k (=5) chunks.

        Pipeline:
        - Embed query
        - FAISS search for a larger candidate pool
        - Optional CrossEncoder rerank
        - Return final top-k chunks
        """

        FINAL_K = int(self.cfg.top_k_retrieve)  # required = 5
        CANDIDATE_K = 30

        qv = self.embed_query(q).astype("float32")[None, :]

        # VectorIndex is a wrapper; the FAISS index is at self.index.index
        D, I = self.index.index.search(qv, CANDIDATE_K)

        candidates: list[dict] = []
        for idx, score in zip(I[0].tolist(), D[0].tolist()):
            if idx == -1:
                continue
            candidates.append(
                {
                    "text": self.index.texts[idx],
                    "metadata": self.index.metadatas[idx],
                    "score": float(score),
                }
            )

        # Optional reranking
        if self.reranker is not None and len(candidates) > 1:
            pairs = [(q, c["text"]) for c in candidates]
            rerank_scores = self.reranker.predict(pairs)
            for c, rs in zip(candidates, rerank_scores):
                c["_rerank_score"] = float(rs)
            candidates.sort(key=lambda x: x["_rerank_score"], reverse=True)
        else:
            candidates.sort(key=lambda x: x["score"], reverse=True)

        top = candidates[:FINAL_K]
        return [{"text": c["text"], "metadata": c["metadata"]} for c in top]

    def answer_question(self, question_obj) -> dict:
        """Answer a question.

        Accepts either a raw string question or a dict
        like {"question_id": int, "question": str}.
        """
        if isinstance(question_obj, dict):
            qid = question_obj.get("question_id")
            q = question_obj.get("question") or question_obj.get("query") or ""
        else:
            qid = None
            q = str(question_obj)

        contexts = self.retrieve(q)

        # Build sources from retrieved chunks (always from top-5)
        sources = []
        for c in contexts:
            md = c.get("metadata", {}) or {}
            sources.append([md.get("document"), md.get("section"), f"p. {md.get('page_start')}"])

        answer_text = generate_json(
            self.tokenizer,
            self.model,
            question=q,
            contexts=contexts,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
        )

        if answer_text in (
            "This question cannot be answered based on the provided documents.",
            "Not specified in the document.",
        ):
            sources = []

        out = {"answer": answer_text, "sources": sources}
        if qid is not None:
            out["question_id"] = qid
        return out
