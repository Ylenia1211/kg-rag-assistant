from __future__ import annotations

from fastapi import APIRouter

from app.schemas.search import (
    SearchEntityResult,
    SearchGraphResult,
    SearchRequest,
    SearchResponse,
    SearchVectorResult,
)
from app.services.search_service import search_knowledge_base

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    result = search_knowledge_base(request.question, limit=request.limit)

    return SearchResponse(
        question=result.question,
        answer=result.answer,
        vector_results=[SearchVectorResult(**item) for item in result.vector_results],
        graph_results=[SearchGraphResult(**item) for item in result.graph_results],
        entity_results=[SearchEntityResult(**item) for item in result.entity_results],
    )