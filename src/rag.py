from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from .config import RagConfig
from .vector_store import VectorIndex, search as faiss_search
from .retrieval import HybridRetriever, hybrid_rank
from .llm import load_llm, build_prompt, generate_json
import json

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



    def answer_question(self, question_obj, fallback_qid=None):
        """
        Robust handler:
        - question_obj can be:
            1) dict: {"question_id": ..., "question": ...}
            2) str containing JSON of that dict
            3) plain str question text
        """

        # --- Normalize question_obj into (qid, question) ---
        if isinstance(question_obj, dict):
            qid = question_obj.get("question_id", fallback_qid)
            question = question_obj.get("question") or question_obj.get("query") or question_obj.get("q")

        elif isinstance(question_obj, str):
            s = question_obj.strip()

            # Case: JSON string line
            if (s.startswith("{") and s.endswith("}")) or (s.startswith('"') and s.endswith('"')):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, dict):
                        qid = parsed.get("question_id", fallback_qid)
                        question = parsed.get("question") or parsed.get("query") or parsed.get("q")
                    else:
                        # JSON loads to a plain string
                        qid = fallback_qid
                        question = str(parsed)
                except Exception:
                    # Not valid JSON, treat as plain question
                    qid = fallback_qid
                    question = question_obj
            else:
                # Plain question text
                qid = fallback_qid
                question = question_obj

        else:
            raise TypeError(f"Unsupported question_obj type: {type(question_obj)}")

        if question is None:
            raise ValueError(f"Could not extract question text from: {question_obj}")

        if qid is None:
            # last resort: allow running without id, but keep deterministic
            qid = fallback_qid if fallback_qid is not None else -1

        # --- Retrieve contexts ---
        contexts = self.retrieve(question)

        # --- Build prompt & generate answer ---
        prompt = build_prompt(question, contexts)

        answer = generate_json(
            self.model,
            self.tokenizer,
            prompt,
            self.cfg.max_new_tokens,
            self.device,
        )

        # --- Enforce rubric strings exactly ---
        a_low = answer.lower()
        if "cannot be answered" in a_low:
            answer = "This question cannot be answered based on the provided documents."
        elif "not specified" in a_low:
            answer = "Not specified in the document."

        # --- Sources ---
        sources = []
        for c in contexts:
            md = c.get("metadata", {}) or {}
            sources.append([
                md.get("document", ""),
                md.get("section", ""),
                f"p. {md.get('page_start')}-{md.get('page_end')}"
            ])

        return {
            "question_id": int(qid) if str(qid).isdigit() else qid,
            "answer": answer,
            "sources": sources
        }