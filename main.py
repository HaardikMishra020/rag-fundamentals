from fastapi import FastAPI, HTTPException,Request,UploadFile, File
from src.ingest import run_ingest_pipeline
from src.schemas import QueryRequest, QueryResponse
from src.query import run_query_pipeline
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from src.store import load_index
from pathlib import Path
import tiktoken


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP — runs once, before any requests are accepted
    app.state.model = SentenceTransformer("all-MiniLM-L6-v2")
    app.state.vectors, app.state.chunks = load_index(Path("data/index"))
    app.state.tokenizer = tiktoken.get_encoding("cl100k_base")
    yield
    # SHUTDOWN — runs once, when the server stops (empty for you, for now)

app = FastAPI(lifespan=lifespan)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request:Request, body_request: QueryRequest):
    try:
        model = request.app.state.model
        vectors = request.app.state.vectors
        chunks = request.app.state.chunks
        answer, _ = run_query_pipeline(body_request.query, model, vectors, chunks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return QueryResponse(answer=answer)


@app.post("/ingest")
async def ingest_endpoint(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename or "uploaded_file.txt"
        with open(Path("data/raw") / filename, "wb") as f:
            f.write(contents)
        
        # Run the ingest pipeline after saving the file
        request.app.state.vectors, request.app.state.chunks = run_ingest_pipeline(
            model=request.app.state.model,
            tokenizer=request.app.state.tokenizer
        )
        return {"status":"success","message": f"File '{filename}' ingested successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest file: {str(e)}")