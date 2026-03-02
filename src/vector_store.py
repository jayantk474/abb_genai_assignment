from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import os
import json
import numpy as np
import faiss

@dataclass
class VectorIndex:
    index: faiss.Index
    texts: List[str]
    metadatas: List[Dict[str, Any]]

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

def search(index: VectorIndex, query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
    q = query_vec.astype(np.float32)
    if q.ndim == 1:
        q = q[None, :]
    scores, ids = index.index.search(q, top_k)
    print("Built index size:", index.ntotal)
    return list(zip(ids[0].tolist(), scores[0].tolist()))
