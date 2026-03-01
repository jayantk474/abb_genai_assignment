from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]

class HybridRetriever:
    """Optional hybrid scorer: combines vector similarity and BM25 for robustness."""
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.corpus_tokens = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def bm25_scores(self, query: str) -> np.ndarray:
        return np.array(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

def normalize_scores(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def hybrid_rank(
    query: str,
    vec_hits: List[Tuple[int, float]],
    bm25: HybridRetriever,
    alpha: float = 0.65,
    top_k: int = 20,
) -> List[Tuple[int, float]]:
    """Combine vector hit list with BM25 into a unified ranking.
    alpha controls weight on vector similarity.
    """
    # Build dense arrays for candidate ids
    cand_ids = [i for i, _ in vec_hits]
    vec_scores = np.array([s for _, s in vec_hits], dtype=np.float32)

    bm25_all = bm25.bm25_scores(query)
    bm_scores = np.array([bm25_all[i] for i in cand_ids], dtype=np.float32)

    vec_n = normalize_scores(vec_scores)
    bm_n = normalize_scores(bm_scores)

    combined = alpha * vec_n + (1 - alpha) * bm_n
    order = np.argsort(-combined)
    ranked = [(cand_ids[int(j)], float(combined[int(j)])) for j in order[:top_k]]
    return ranked
