from dataclasses import dataclass

@dataclass
class RagConfig:
    # Embeddings
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Reranker (CrossEncoder). Set to "" to disable.
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # LLM (open) – small enough for Colab CPU/GPU; works better on GPU.
    llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # Chunking
    chunk_chars: int = 3500
    chunk_overlap_chars: int = 400
    # Retrieval
    top_k_retrieve: int = 20
    top_k_rerank: int = 5
    # Answer controls
    max_new_tokens: int = 450
    temperature: float = 0.2
