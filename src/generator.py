from typing import List, Tuple
from groq import Groq
from src.chunker import Chunk
from src.config import GROQ_API_KEY

def generate_answer(query: str, retrieved_chunks: List[Tuple[Chunk, float]], model_name: str = "llama-3.1-8b-instant") -> Tuple[str,List[Chunk]]:
    "Build a prompt from a query and retrieved chunks, call an LLM API, and return the answer."
    response = ""
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. If the answer is not contained within the context, respond with 'I don't know.'"
            },
            {
                "role": "user",
                "content": f"Answer the following question based on the provided context:\n\nQuestion: {query}\n\nContext:\n" + "\n".join([chunk.content + " (" + chunk.filename + ")" for chunk, _ in retrieved_chunks])
            }
        ],
        temperature=0.3,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None
    )

    for res_chunk in completion:
        response += res_chunk.choices[0].delta.content or ""

    return response, [chunk for chunk, score in retrieved_chunks]
