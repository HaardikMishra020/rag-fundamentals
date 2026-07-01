from pathlib import Path
from src.embedder import embed_query
from src.store import load_index, search_index
from src.generator import generate_answer
from sentence_transformers import SentenceTransformer
import json

if __name__ == "__main__":
    # Wire embedder -> store -> generator to answer a question.
    with open(Path("test/questions.json")) as f:
        questions = json.load(f)
    
    for question in questions:
        input_query = question["question"]
        print(f"Question: {input_query}")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embed_query_vector = embed_query(input_query, model=model)
        load_index_vectors, load_index_chunks = load_index(Path("data/index"))
        retrieved_chunks = search_index(embed_query_vector, load_index_vectors, load_index_chunks, top_k=5)
        # for chunk, score in retrieved_chunks:
        #     print(f"Retrieved Chunk: (Index: {chunk.index}) with Score: {score:.4f}")
        if max([score for _, score in retrieved_chunks]) > 0.01:    
            answer, used_chunks = generate_answer(input_query, retrieved_chunks, model_name="llama-3.1-8b-instant")
        else:
            answer, used_chunks = "I don't know.", []
        print("Answer:", answer)
        print("Used Chunks:", [f"{chunk.filename}: {chunk.index}" for chunk in used_chunks])
