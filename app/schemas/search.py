from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User search question")
    limit: int | None = Field(default=None, ge=1, le=20)


class SearchVectorResult(BaseModel):
    id: str
    score: float
    payload: dict


class SearchGraphResult(BaseModel):
    document_id: str
    chunk_id: int
    text: str
    start_char: int
    end_char: int


class SearchEntityResult(BaseModel):
    document_id: str
    entity_id: str
    entity_name: str
    entity_type: str


class SearchResponse(BaseModel):
    question: str
    answer: str
    vector_results: list[SearchVectorResult] = Field(default_factory=list)
    graph_results: list[SearchGraphResult] = Field(default_factory=list)
    entity_results: list[SearchEntityResult] = Field(default_factory=list)