from dataclasses import dataclass

@dataclass
class RagConfig:
    # Embeddings
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Reranker (CrossEncoder). Set to "" to disable.
    reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # LLM (open) – small enough for Colab CPU/GPU; works better on GPU.
    llm_model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    max_new_tokens: int = 128
    temperature: float = 0.0
    # Chunking
    # 10-Ks contain long tables; small chunks often split the key row/number.
    # Use larger chunks with modest overlap to keep table rows intact.
    chunk_chars: int = 3500
    chunk_overlap_chars: int = 400

    # Retrieval
    top_k_retrieve = 5
    # Candidates kept after hybrid scoring before cross-encoder rerank
    top_k_rerank = 30
