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
    ap.add_argument("--out_dir", default="artifacts/index", help="Where to save the index root directory")
    args = ap.parse_args()

    cfg = RagConfig()
    embedder = SentenceTransformer(cfg.embed_model_name)

    def build_one(pdf_path: str, doc_name: str, out_dir: str):
        pages = extract_pdf_pages(pdf_path)
        chunks = list(chunk_pages(pages, doc_name, cfg.chunk_chars, cfg.chunk_overlap_chars))

        texts = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        print(f"{doc_name}: chunks={len(texts)}")

        embs = []
        for i in tqdm(range(0, len(texts), 64), desc=f"Embedding {doc_name}"):
            batch = texts[i:i+64]
            e = embedder.encode(batch, normalize_embeddings=True)
            embs.append(e)
        embs = np.vstack(embs).astype(np.float32)

        idx = build_faiss(embs)
        store = VectorIndex(index=idx, texts=texts, metadatas=metas)
        store.save(out_dir)

    apple_out = os.path.join(args.out_dir, "apple")
    tesla_out = os.path.join(args.out_dir, "tesla")
    os.makedirs(apple_out, exist_ok=True)
    os.makedirs(tesla_out, exist_ok=True)

    build_one(args.apple_pdf, "Apple 10-K", apple_out)
    build_one(args.tesla_pdf, "Tesla 10-K", tesla_out)

    # A tiny manifest so scripts can detect multi-index mode.
    manifest = {
        "apple": {"dir": "apple", "document": "Apple 10-K"},
        "tesla": {"dir": "tesla", "document": "Tesla 10-K"},
    }
    import json
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved indices to: {args.out_dir} (apple/, tesla/)")

if __name__ == "__main__":
    main()
