from __future__ import annotations

from pydantic import BaseModel, Field


class UploadedDocumentInfo(BaseModel):
    filename: str = Field(..., description="Original uploaded filename")
    content_type: str = Field(..., description="Detected MIME type")
    size_bytes: int = Field(..., ge=0, description="Uploaded file size in bytes")
    parsed: bool = Field(..., description="Whether text parsing succeeded")
    text_length: int = Field(..., ge=0, description="Extracted text length in characters")
    preview: str = Field(..., description="Short preview of extracted text")
    chunk_count: int = Field(..., ge=0, description="Number of generated text chunks")
    stored_chunk_count: int = Field(..., ge=0, description="Number of chunks stored in Qdrant")
    graph_chunk_count: int = Field(..., ge=0, description="Number of chunks stored in Neo4j")
    entity_count: int = Field(..., ge=0, description="Number of extracted entities")
    relation_count: int = Field(..., ge=0, description="Number of inferred relations")
    document_id: str | None = Field(default=None, description="Generated logical document ID")
    error: str | None = Field(default=None, description="Parsing or chunking error, if any")


class DocumentUploadResponse(BaseModel):
    uploaded: int = Field(..., ge=0)
    files: list[UploadedDocumentInfo] = Field(default_factory=list)
    message: str = Field(...)