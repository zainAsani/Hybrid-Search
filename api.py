"""A small web API around the search engine, built with FastAPI.

Start it:   uvicorn api:app --reload
Then open:  http://127.0.0.1:8000/docs   (FastAPI creates this test page for you)
"""
from contextlib import asynccontextmanager
from typing import List, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from search.data import load_products
from search.engine import SearchEngine

engine = None  # created once at start-up, then reused for every request


@asynccontextmanager
async def lifespan(app):
    # Code before "yield" runs once when the server starts.
    # Loading the models is slow, so we do it here and not on every request.
    global engine
    engine = SearchEngine(load_products())
    yield
    # Code after "yield" would run when the server shuts down (nothing needed here).


app = FastAPI(title="Hybrid Product Search", lifespan=lifespan)


# Pydantic models describe what JSON goes in and out. FastAPI uses them to
# check the input automatically and to build the /docs page.
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["earphones for jogging"])
    top_k: int = Field(5, ge=1, le=20)  # ge = greater or equal, le = less or equal
    method: Literal["bm25", "dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank"


class SearchHit(BaseModel):
    id: str
    title: str
    score: float


class SearchResponse(BaseModel):
    query: str
    method: str
    results: List[SearchHit]


@app.get("/health")
def health():
    """Simple check that the service is running (used by Docker, cloud platforms, monitoring)."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    hits = engine.search(request.query, method=request.method, top_k=request.top_k)
    results = [
        SearchHit(id=doc_id, title=engine.products_by_id[doc_id]["title"], score=score)
        for doc_id, score in hits
    ]
    return SearchResponse(query=request.query, method=request.method, results=results)
