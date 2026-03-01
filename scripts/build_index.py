from __future__ import annotations
import os
import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.config import RagConfig
from src.pdf_utils import extract_pdf_pages, chunk_pages
from src.vector_store import VectorIndex, build_faiss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apple_pdf", required=True, help="Path to Apple 2024 10-K PDF")
    ap.add_argument("--tesla_pdf", required=True, help="Path to Tesla 2023 10-K PDF")
    ap.add_argument("--out_dir", default="artifacts/index", help="Where to save the index")
    args = ap.parse_args()

    cfg = RagConfig()
    embedder = SentenceTransformer(cfg.embed_model_name)

    chunks = []
    # Apple
    apple_pages = extract_pdf_pages(args.apple_pdf)
    for ch in chunk_pages(apple_pages, "Apple 10-K", cfg.chunk_chars, cfg.chunk_overlap_chars):
        chunks.append(ch)
    # Tesla
    tesla_pages = extract_pdf_pages(args.tesla_pdf)
    for ch in chunk_pages(tesla_pages, "Tesla 10-K", cfg.chunk_chars, cfg.chunk_overlap_chars):
        chunks.append(ch)

    texts = [c["text"] for c in chunks]
    metas = [c["metadata"] for c in chunks]

    print(f"Total chunks: {len(texts)}")
    embs = []
    for i in tqdm(range(0, len(texts), 64), desc="Embedding"):
        batch = texts[i:i+64]
        e = embedder.encode(batch, normalize_embeddings=True)
        embs.append(e)
    embs = np.vstack(embs).astype(np.float32)

    idx = build_faiss(embs)
    store = VectorIndex(index=idx, texts=texts, metadatas=metas)
    store.save(args.out_dir)
    print(f"Saved index to: {args.out_dir}")

if __name__ == "__main__":
    main()
