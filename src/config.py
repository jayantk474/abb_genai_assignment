from dataclasses import dataclass

@dataclass
class RagConfig:
    # Embeddings
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Reranker (CrossEncoder). Set to "" to disable.
    reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_model_name = "Qwen/Qwen2-1.5B-Instruct"
    max_new_tokens: int = 120
    temperature: float = 0.0
    # Chunking
    chunk_chars: int = 1200
    chunk_overlap_chars: int = 150

    # Retrieval
    top_k_retrieve: int = 5
    top_k_rerank: int = 3

