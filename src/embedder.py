from typing import List
import numpy as np
from src.chunker import Chunk

def embed_chunks(chunks: List[Chunk], model=None) -> np.ndarray:
    "Convert a list of texts into vectors using a sentence-transformers model."
    if not chunks:
        raise ValueError("Input chunk list is empty.")
    if not model:
        raise ValueError("Model is not provided.")
    
    chunk_content_list = []
    for chunk in chunks:
        if not isinstance(chunk, Chunk):
            raise TypeError("Each item in chunks must be an instance of Chunk.")
        if not chunk.content:
            raise ValueError("Chunk content cannot be empty.")
        
        chunk_content_list.append(chunk.content)
    
    embeddings = model.encode(chunk_content_list, convert_to_numpy=True, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    return embeddings

def embed_query(query: str, model=None) -> np.ndarray:
    "Convert a query string into a vector using a sentence-transformers model."
    if not query:
        raise ValueError("Input query string is empty.")
    if not model:
        raise ValueError("Model is not provided.")
    
    query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return query_vector[0]

