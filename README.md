rag-fundamentals/
├── data/
│   └── raw/                  
├── src/
│   ├── __init__.py
│   ├── loader.py              
│   ├── chunker.py             
│   ├── embedder.py            
│   ├── store.py                
│   ├── generator.py            
│   ├── ingest.py               
│   └── query.py                 
├── test/
│   └── questions.json          
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md

File responsibilities (one-line each):
- `data/raw/`: directory to place raw input text files used for ingestion.
- `src/__init__.py`: package marker for the `src` module.
- `src/loader.py`: load raw text files from a directory into a list of strings.
- `src/chunker.py`: split text into fixed-size overlapping chunks.
- `src/embedder.py`: convert text (chunks or queries) into vectors using a sentence-transformers model.
- `src/store.py`: save/load chunk vectors and their source text, and search for nearest neighbors to a query vector.
- `src/generator.py`: take a query and retrieved chunks, build a prompt, call an LLM API, return the answer.
- `src/ingest.py`: script that wires loader -> chunker -> embedder -> store to build the index (entry-point only).
- `src/query.py`: script that wires embedder -> store -> generator to answer a question (entry-point only).
- `test/questions.json`: placeholder for evaluation questions (not populated).
- `.env.example`: placeholder environment variables for API keys.
- `requirements.txt`: minimal libraries required by the project.

# Running the Project
 
## 1. Install dependencies
 
```bash
pip install -r requirements.txt
```

 
## 2. Set up your API key
 
Create a `.env` file in the project root (`rag-fundamentals/.env`) similar to `.env.example`:
 
 
## 3. Add your source document(s)
 
Place your `.txt` / `.md` files here:
 
```
data/raw/
```
 
e.g. `data/raw/nimbus_handbook.txt`
 
## 4. Run ingestion (builds the index — do this first, and any time your source docs change)
 
```bash
cd rag-fundamentals
python -m src.ingest
```
 
Check that it created:
 
```
data/index/embeddings.npy
data/index/chunks.pkl
```
 
## 5. Run a query
 
```bash
python -m src.query
```
 
 
## 6. What to check in the output
 
- **Answer** — is it factually correct against the source doc?
- **Used Chunks** — which chunks were retrieved and passed to the LLM (filename + chunk index), useful for debugging retrieval quality
- If you get an authentication error from Groq — check step 2 (API key / `load_dotenv()`), not the RAG logic
- If `query.py` fails to find the index — check step 4 ran successfully first
## Notes
 
- Similarity threshold in `query.py` is currently set very low (~0.01) — it's not doing meaningful filtering. The system prompt's grounding instruction ("say I don't know if not in context") is what's actually preventing hallucinated answers, not the threshold. See eval results in project context for why.
- Run `test/questions.json` questions manually through `query.py` to test retrieval/generation quality across categories (easy lookup, precision, multi-hop, distractor, out-of-scope).
 