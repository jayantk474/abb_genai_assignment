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

        qv = self.embedder.encode([q], normalize_embeddings=True)
        qv = np.array(qv).astype("float32")

        # ---- FAISS search (correct wrapper access) ----
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

        # ---- IMPORTANT: Correct sorting ----
        # If index built with cosine/IP → higher is better
        # Most MiniLM FAISS builds use IndexFlatIP
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # ---- Optional reranker ----
        if self.reranker is not None and len(candidates) > 1:
            pairs = [(q, c["text"]) for c in candidates]
            rerank_scores = self.reranker.predict(pairs)

            for c, rs in zip(candidates, rerank_scores):
                c["_rerank_score"] = float(rs)

            candidates.sort(key=lambda x: x["_rerank_score"], reverse=True)

        top = candidates[:FINAL_K]

        return [
            {"text": c["text"], "metadata": c["metadata"]}
            for c in top
        ]

    def answer_question(self, question_obj):

        qid = question_obj["question_id"]
        question = question_obj["question"]

        contexts = self.retrieve(question)

        prompt = build_prompt(question, contexts)

        answer = generate_json(
            self.model,
            self.tokenizer,
            prompt,
            self.cfg.max_new_tokens,
            self.device,
        )

        # ---- Enforce rubric strings exactly ----
        if "cannot be answered" in answer.lower():
            answer = "This question cannot be answered based on the provided documents."

        if "not specified" in answer.lower():
            answer = "Not specified in the document."

        # ---- Build source list ----
        sources = []
        for c in contexts:
            md = c["metadata"]
            sources.append([
                md.get("document", ""),
                md.get("section", ""),
                f"p. {md.get('page_start')}-{md.get('page_end')}"
            ])

        return {
            "question_id": qid,
            "answer": answer,
            "sources": sources
        }
