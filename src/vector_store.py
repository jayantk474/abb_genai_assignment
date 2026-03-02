from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Union
import os
import json
import numpy as np
import faiss

@dataclass
class VectorIndex:
    index: faiss.Index
    texts: List[str]
    metadatas: List[Dict[str, Any]]

    @property
    def ntotal(self) -> int:
        """Compatibility shim for code that expects `VectorIndex.ntotal` like FAISS."""
        return int(getattr(self.index, "ntotal", 0))

    def __len__(self) -> int:
        return len(self.texts)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(out_dir, "faiss.index"))
        with open(os.path.join(out_dir, "store.jsonl"), "w", encoding="utf-8") as f:
            for t, m in zip(self.texts, self.metadatas):
                f.write(json.dumps({"text": t, "metadata": m}, ensure_ascii=False) + "\n")

    @staticmethod
    def load(in_dir: str) -> "VectorIndex":
        idx = faiss.read_index(os.path.join(in_dir, "faiss.index"))
        texts, metas = [], []
        with open(os.path.join(in_dir, "store.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj["text"])
                metas.append(obj["metadata"])
        return VectorIndex(index=idx, texts=texts, metadatas=metas)

def build_faiss(embeddings: np.ndarray) -> faiss.Index:
    # Cosine similarity via inner product on normalized vectors
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index

def search(index: Union[VectorIndex, faiss.Index], query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
    """Vector similarity search.

    Accepts either the repo's `VectorIndex` wrapper or a raw `faiss.Index`.
    Returns a list of (doc_id, score) pairs and filters out missing ids (-1).
    """
    q = query_vec.astype(np.float32)
    if q.ndim == 1:
        q = q[None, :]

    faiss_index = index.index if hasattr(index, "index") else index
    scores, ids = faiss_index.search(q, top_k)

    out: List[Tuple[int, float]] = []
    for doc_id, score in zip(ids[0].tolist(), scores[0].tolist()):
        if int(doc_id) == -1:
            continue
        out.append((int(doc_id), float(score)))
    return out
