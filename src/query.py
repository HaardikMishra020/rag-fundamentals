from pathlib import Path
import numpy as np
from typing import List, Tuple
from src.embedder import embed_query
from src.store import load_index, search_index
from src.generator import generate_answer
from sentence_transformers import SentenceTransformer
from src.chunker import Chunk
import json


def run_query_pipeline(query: str, model:SentenceTransformer, vectors:np.ndarray, chunks:List[Chunk]) -> Tuple[str, List[Chunk]]:    
    "Run the embed -> search -> generate pipeline for a single query and return (answer, used_chunks)."
    query_vector = embed_query(query, model=model)
    retrieved = search_index(query_vector, vectors, chunks, top_k=5)
    if not retrieved:
        return "I don't know.", []
    
    answer, used_chunks = generate_answer(query, retrieved, model_name="llama-3.1-8b-instant")
    return answer, used_chunks


if __name__ == "__main__":

    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the embedding model
    vectors, chunks = load_index(Path("data/index"))
    # Wire embedder -> store -> generator to answer a question.
    with open(Path("test/questions.json")) as f:
        questions = json.load(f)
    
    for question in questions:
        input_query = question["question"]
        print(f"Question: {input_query}")
        answer, used_chunks = run_query_pipeline(input_query,model,vectors,chunks)
        print("Answer:", answer)
        print("Used Chunks:", [f"{chunk.filename}: {chunk.index}" for chunk in used_chunks])
