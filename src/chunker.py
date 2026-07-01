from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Chunk:
    filename: str
    content: str
    index: int


def chunk_text(text: List[Tuple[str, str]], chunk_size: int = 200, overlap: int = 50, tokenizer=None) -> List[Chunk]:
    "Split text into fixed-size overlapping chunks."
    chunks = []
    if not text:
        raise ValueError("Input text list is empty.")
    
    if not tokenizer:
        raise ValueError("Tokenizer is not provided.")
    
    for filename, content in text:
        if not isinstance(filename, str) or not isinstance(content, str):
            raise TypeError("Each item in text must be a tuple of (filename: str, content: str).")
        if not filename or not content:
            raise ValueError("Filename and content cannot be empty.")
        
        tokens = tokenizer.encode(content)
        i=0
        for token_start in range(0, len(tokens), chunk_size - overlap):
            token_end = min(token_start + chunk_size, len(tokens))
            chunk_tokens = tokens[token_start:token_end]
            chunk_content = tokenizer.decode(chunk_tokens)
            chunks.append(Chunk(filename, chunk_content, i))
            i += 1
            if(token_end == len(tokens)):
                break
    return chunks
