# rag-fundamentals

A RAG (Retrieval-Augmented Generation) system built from scratch to learn the
core mechanics of retrieval, chunking, embedding, and grounded generation —
no framework abstraction, every line written and defensible.

## Folder Structure

```
rag-fundamentals/
├── data/
│   ├── raw/                   # source documents (.txt)
│   └── index/                 # persisted embeddings.npy + chunks.pkl
├── src/
│   ├── __init__.py
│   ├── loader.py               # load raw text files from a directory
│   ├── chunker.py               # split text into overlapping chunks
│   ├── embedder.py               # text -> vectors (chunks + queries)
│   ├── store.py                   # save/load index, nearest-neighbor search
│   ├── generator.py                # query + chunks -> prompt -> LLM answer
│   ├── ingest.py                    # loader -> chunker -> embedder -> store
│   ├── query.py                      # embedder -> store -> generator
│   ├── schemas.py                     # Pydantic request/response models
│   └── config.py                       # centralized .env loading, API keys
├── test/
│   └── questions.json                  # hand-written eval questions
├── main.py                               # FastAPI app (/query, /ingest)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## File Responsibilities

- `data/raw/` — directory for source documents used at ingestion.
- `data/index/` — persisted vector index (`embeddings.npy`, `chunks.pkl`), rebuilt on every ingest.
- `src/loader.py` — load raw text files from a directory into memory.
- `src/chunker.py` — split text into fixed-size overlapping chunks (`Chunk` dataclass).
- `src/embedder.py` — convert text (chunks or a single query) into normalized vectors via `all-MiniLM-L6-v2`.
- `src/store.py` — save/load the index to/from disk, search for nearest neighbors via dot product.
- `src/generator.py` — build a grounded prompt from retrieved chunks, call the Groq LLM, return `(answer, used_chunks)`.
- `src/ingest.py` — `run_ingest_pipeline()`: rebuilds the full index from everything in `data/raw/`. Also runnable standalone (`python -m src.ingest`).
- `src/query.py` — `run_query_pipeline()`: runs embed → search → generate for one query. Also runnable standalone (`python -m src.query`), loops over `test/questions.json`.
- `src/schemas.py` — `QueryRequest`/`QueryResponse` Pydantic models for the API layer.
- `src/config.py` — single place that calls `load_dotenv()` and exposes `GROQ_API_KEY`; every other module imports from here instead of reading `os.environ` directly.
- `main.py` — FastAPI app. Loads the embedding model + index once at startup (`lifespan`, stored in `app.state`), exposes `POST /query` and `POST /ingest`.
- `test/questions.json` — 18 eval questions across categories: easy_lookup, precision_numeric, precision_name, multi_hop, distractor, out_of_scope.

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up your API key

Create a `.env` file in the project root (`rag-fundamentals/.env`), same shape as `.env.example`. `src/config.py` loads this automatically — no other file reads `os.environ` directly.

### 3. Add your source document(s)

Place `.txt` files in `data/raw/`, e.g. `data/raw/nimbus_handbook.txt`.

### 4. Build the index

Either the CLI script or the API endpoint works — both call the same `run_ingest_pipeline()`.

**CLI:**
```bash
python -m src.ingest
```

**API** (server must be running — see step 5):
```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "file=@data/raw/your_new_doc.txt"
```

Either way, confirm `data/index/embeddings.npy` and `data/index/chunks.pkl` were created/updated.

### 5. Run a query

**CLI** (loops through `test/questions.json`):
```bash
python -m src.query
```

**API:**
```bash
uvicorn main:app --reload
```
Then either `curl` `POST /query` with `{"query": "..."}`, or use the Swagger UI at `http://127.0.0.1:8000/docs` (also the easiest way to test file uploads for `/ingest`).

### 6. What to check in the output

- **Answer** — factually correct against the source doc(s)?
- **Used Chunks** (CLI only) — which chunks were retrieved (filename + index) — useful for debugging retrieval quality.
- Groq auth error → check step 2 (`.env` / `config.py`), not the RAG logic.
- `query.py`/`/query` can't find the index → check step 4 ran successfully first.
- `/ingest` returns 500 → check the error detail in the response; the file may have failed to write or the pipeline may have hit a bad input.

## Notes

- Similarity threshold in the retrieval step is set very low (~0.01) — it's not doing meaningful filtering. The generator's system prompt ("say I don't know if not in context") is what actually prevents hallucinated answers, not the threshold — verified against all out-of-scope/distractor eval questions.
- `test/questions.json` questions can be run manually through `query.py` (CLI) or `/query` (API) to spot-check retrieval/generation quality across categories.

## Session Log

### Phase 2 — FastAPI wrapper (this session)

Wrapped the existing CLI pipelines in a FastAPI app with two endpoints:

- **`POST /query`** — `{"query": str}` → `{"answer": str}`. Empty/whitespace queries now return `400` (fixed a bug where `[""]` slipped past `encode()`'s validation). `ValueError` → `400`, everything else → `500`.
- **`POST /ingest`** — accepts a file upload, saves it to `data/raw/`, rebuilds the full index, refreshes in-memory state — no restart needed.
- **Startup/state**: model + index load once at startup via `lifespan`, cached on `app.state`, instead of reloading per request — the main architectural fix this session.
- Centralized `.env` loading into `src/config.py`.
- **Verified end-to-end**: ingested a new doc live, confirmed `/query` answered correctly from both the new and original corpus.
