from pathlib import Path
import tiktoken
from src.chunker import Chunk, chunk_text
from src.embedder import embed_chunks
from src.loader import load_raw_texts
from src.store import save_index
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple


# Load the model

def run_ingest_pipeline(model: SentenceTransformer, tokenizer) -> Tuple[np.ndarray, List[Chunk]]:
    "Run the loader -> chunker -> embedder -> store pipeline to build the index."
    # encoding = tiktoken.get_encoding("cl100k_base")
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    loaded_texts = load_raw_texts(Path("data/raw"))
    chunks = chunk_text(loaded_texts, chunk_size=200, overlap=20, tokenizer=tokenizer)
    embeddings = embed_chunks(chunks, model=model)
    save_index(embeddings, chunks, Path("data/index"))
    return (embeddings, chunks)


if __name__ == "__main__":
    # Wire loader -> chunker -> embedder -> store to build the index.

    encoding = tiktoken.get_encoding("cl100k_base")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    run_ingest_pipeline(model=model, tokenizer=encoding)    
    
