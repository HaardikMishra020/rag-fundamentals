from pathlib import Path
from typing import List, Tuple

def load_raw_texts(directory: Path) -> List[Tuple[str, str]]:
    "Load raw text files from a directory into a list of (filename, content) tuples."
    file_paths = list(directory.glob("*.txt")) + list(directory.glob("*.md"))
    if not file_paths:
        raise ValueError(f"No text files found in directory: {directory}")
    texts = []
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                texts.append((file_path.name, content))
        except Exception as e:
            raise IOError(f"Error reading file {file_path}: {e}")
    return texts
