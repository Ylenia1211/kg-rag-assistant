from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.documents import DocumentUploadResponse, UploadedDocumentInfo
from app.services.ingestion_service import ingest_document

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> DocumentUploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    uploaded_files: list[UploadedDocumentInfo] = []

    for file in files:
        content = await file.read()
        content_type = file.content_type or "application/octet-stream"

        if content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {content_type}",
            )

        size_bytes = len(content)
        if size_bytes > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {file.filename}",
            )

        result = ingest_document(
            filename=file.filename or "unknown",
            content_type=content_type,
            content=content,
        )

        uploaded_files.append(
            UploadedDocumentInfo(
                filename=result.filename,
                content_type=result.content_type,
                size_bytes=result.size_bytes,
                parsed=result.parsed,
                text_length=result.text_length,
                preview=result.preview,
                chunk_count=len(result.chunks),
                stored_chunk_count=result.stored_chunk_count,
                graph_chunk_count=result.graph_chunk_count,
                entity_count=result.entity_count,
                relation_count=result.relation_count,
                document_id=result.document_id,
                error=result.error,
            )
        )

        await file.close()

    return DocumentUploadResponse(
        uploaded=len(uploaded_files),
        files=uploaded_files,
        message="Files received, parsed, chunked, stored in Qdrant, and linked in Neo4j.",
    )