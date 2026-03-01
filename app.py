from __future__ import annotations
# Public interface required by the assignment.
from src.config import RagConfig
from src.vector_store import VectorIndex
from src.rag import RagSystem

# Lazy singleton cache
_RAG = None

def _get_rag(index_dir: str = "artifacts/index") -> RagSystem:
    global _RAG
    if _RAG is None:
        cfg = RagConfig()
        index = VectorIndex.load(index_dir)
        _RAG = RagSystem(cfg, index)
    return _RAG

def answer_question(query: str) -> dict:
    """Return a dict with keys: answer, sources.
    sources is a list like: ["Apple 10-K", "Item 8", "p. 282"].
    """
    rag = _get_rag()
    return rag.answer_question(query)
