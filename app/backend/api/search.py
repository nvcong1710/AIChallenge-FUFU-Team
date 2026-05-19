from functools import lru_cache

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...common.config import get_config
from ..services.search_engine import SearchEngine

router = APIRouter(prefix="/api", tags=["search"])


@lru_cache(maxsize=1)
def get_search_engine() -> SearchEngine:
    return SearchEngine(get_config())


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(20, ge=1, le=100)


@router.post("/search")
def search(req: SearchRequest):
    try:
        return get_search_engine().search(req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
def stats():
    try:
        return get_search_engine().retriever.stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
