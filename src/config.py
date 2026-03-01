from dataclasses import dataclass

@dataclass
class RagConfig:
    # Embeddings
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Reranker (CrossEncoder). Set to "" to disable.
    reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # LLM (open) – small enough for Colab CPU/GPU; works better on GPU.
    llm_model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    max_new_tokens: int = 120
    temperature: float = 0.0
    # Chunking
    chunk_chars: int = 1200
    chunk_overlap_chars: int = 150

    # Retrieval
    top_k_retrieve: int = 5
    top_k_rerank: int = 5

