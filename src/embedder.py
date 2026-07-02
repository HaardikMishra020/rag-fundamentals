from typing import List
import numpy as np
from src.chunker import Chunk

def embed_chunks(chunks: List[Chunk], model=None) -> np.ndarray:
    "Convert a list of texts into vectors using a sentence-transformers model."
    
    chunk_content_list = []
    for chunk in chunks:
        if not isinstance(chunk, Chunk):
            raise TypeError("Each item in chunks must be an instance of Chunk.")
        if not chunk.content:
            raise ValueError("Chunk content cannot be empty.")
        
        chunk_content_list.append(chunk.content)
    
    embeddings = encode(chunk_content_list, model=model, show_progress_bar=True)
    return embeddings


def embed_query(query: str, model=None) -> np.ndarray:
    "Convert a query string into a vector using a sentence-transformers model."    
    query_vector = encode([query], model=model, show_progress_bar=False)
    return query_vector[0]


def encode(data:List[str],model=None, show_progress_bar=False) -> np.ndarray:
    "Convert a list of strings into vectors using a sentence-transformers model."
    if not data or not isinstance(data,list) or len(data) == 0 or any(d.strip() == "" for d in data):
        print("Data is empty or not a list.")
        raise ValueError("Input data list is empty.")
    if not model:
        raise ValueError("Model is not provided.")
    embeddings = model.encode(data, convert_to_numpy=True, show_progress_bar=show_progress_bar, batch_size=32, normalize_embeddings=True)
    print(f"Encoded {len(data)} items into embeddings with shape: {embeddings.shape}")
    return embeddings

