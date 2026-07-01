from pathlib import Path
from typing import List, Tuple
import numpy as np
import pickle
from src.chunker import Chunk

def save_index(vectors: np.ndarray, chunks: List[Chunk],path:Path) -> None:
    "Save chunk vectors and their source texts to disk."
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "embeddings.npy", vectors)
    with open(path / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    return None

def load_index(path: Path) -> Tuple[np.ndarray, List[Chunk]]:
    "Load chunk vectors and their source texts from disk."
    if not path.exists():
        raise FileNotFoundError(f"Index directory does not exist: {path}")
    
    vectors = np.load(path / "embeddings.npy")
    with open(path / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return vectors, chunks

def search_index(query_vector: np.ndarray, vectors: np.ndarray, chunks: List[Chunk], top_k: int = 5) -> List[Tuple[Chunk, float]]:
    "Search for nearest neighbors to a query vector and return top-k texts with scores."
    if query_vector.ndim != 1:
        raise ValueError("Query vector must be a 1D array.")
    if vectors.ndim != 2:
        raise ValueError("Vectors must be a 2D array.")
    if len(chunks) != vectors.shape[0]:
        raise ValueError("Number of chunks must match number of vectors.")
    
    scores = vectors @ query_vector
    top_k_scores = sorted(np.argpartition(scores, -top_k)[-top_k:], key=lambda i: scores[i], reverse=True)
    top_k_chunks = [(chunks[i], scores[i]) for i in top_k_scores]
    return top_k_chunks
