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

    def retrieve(self, q: str) -> List[Dict[str, Any]]:
        qv = self.embed_query(q)
        vec_hits = faiss_search(self.index, qv, self.cfg.fetch_k)
        # Hybrid combine (vector + BM25) for robustness
        ranked = hybrid_rank(q, vec_hits, self.bm25, alpha=0.25, top_k=self.cfg.fetch_k)

        # Prepare candidates
        candidates = [{
            "text": self.index.texts[i],
            "metadata": self.index.metadatas[i],
            "score": s,
            "id": i
        } for i, s in ranked if i != -1]

        # Rerank with CrossEncoder for final top_k_rerank
        if self.reranker and candidates:
            pairs = [(q, c["text"]) for c in candidates]
            rr = self.reranker.predict(pairs)
            for c, r in zip(candidates, rr):
                c["rerank_score"] = float(r)
            candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        top = candidates[: self.cfg.top_k_retrieve]  # final top-5 after rerank
        return [{"text": c["text"], "metadata": c["metadata"]} for c in top]

    def answer_question(self, q: str) -> Dict[str, Any]:
        contexts = self.retrieve(q)
        prompt = build_prompt(q, contexts)
        obj = generate_json(
            self.tokenizer,
            self.model,
            prompt,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
        )

        # Ensure sources are well-formed and come from retrieved metadata.
        # We'll map to citations based on chosen chunks (avoid hallucinated citations).
        if obj["answer"] in [
            "This question cannot be answered based on the provided documents.",
            "Not specified in the document."
        ]:
            obj["sources"] = []
            return obj

        # Otherwise, attach citations from the retrieved chunks (top-5)
        sources = []
        for c in contexts[:5]:
            md = c["metadata"]
            sources.append([md["document"], md["section"], f"p. {md['page_start']}"])
        obj["sources"] = sources
        print("Retrieved", len(contexts), "chunks")
        print(contexts[0]["metadata"])
        print(contexts[0]["text"][:400])
        return obj
