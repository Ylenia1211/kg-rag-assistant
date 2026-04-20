from __future__ import annotations

from fastapi.testclient import TestClient


def test_upload_documents_success(monkeypatch):
    import app.main as main_module
    import app.api.routes.documents as documents_module
    from app.services.ingestion_service import IngestionResult
    from app.services.text_chunker import TextChunk

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "ensure_qdrant_collection", lambda: None)
    monkeypatch.setattr(main_module, "ensure_neo4j_constraints", lambda: None)
    monkeypatch.setattr(main_module, "close_neo4j_driver", lambda: None)

    monkeypatch.setattr(
        documents_module,
        "ingest_document",
        lambda filename, content_type, content: IngestionResult(
            filename=filename,
            content_type=content_type,
            size_bytes=len(content),
            parsed=True,
            text_length=11,
            preview="hello world",
            error=None,
            chunks=[
                TextChunk(chunk_id=1, text="hello worl", start_char=0, end_char=10),
                TextChunk(chunk_id=2, text="rld", start_char=8, end_char=11),
            ],
            document_id="doc-123",
            stored_chunk_count=2,
            graph_chunk_count=2,
            entity_count=2,
            relation_count=1,
        ),
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/v1/documents/upload",
            files=[("files", ("note.txt", b"hello world", "text/plain"))],
        )

    assert response.status_code == 200
    assert response.json() == {
        "uploaded": 1,
        "files": [
            {
                "filename": "note.txt",
                "content_type": "text/plain",
                "size_bytes": 11,
                "parsed": True,
                "text_length": 11,
                "preview": "hello world",
                "chunk_count": 2,
                "stored_chunk_count": 2,
                "graph_chunk_count": 2,
                "entity_count": 2,
                "relation_count": 1,
                "document_id": "doc-123",
                "error": None,
            }
        ],
        "message": "Files received, parsed, chunked, stored in Qdrant, and linked in Neo4j.",
    }


def test_upload_documents_reports_ingestion_failure(monkeypatch):
    import app.main as main_module
    import app.api.routes.documents as documents_module
    from app.services.ingestion_service import IngestionResult

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "ensure_qdrant_collection", lambda: None)
    monkeypatch.setattr(main_module, "ensure_neo4j_constraints", lambda: None)
    monkeypatch.setattr(main_module, "close_neo4j_driver", lambda: None)

    monkeypatch.setattr(
        documents_module,
        "ingest_document",
        lambda filename, content_type, content: IngestionResult(
            filename=filename,
            content_type=content_type,
            size_bytes=len(content),
            parsed=False,
            text_length=0,
            preview="",
            error="Unable to parse PDF file",
            chunks=[],
            document_id=None,
            stored_chunk_count=0,
            graph_chunk_count=0,
            entity_count=0,
            relation_count=0,
        ),
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/v1/documents/upload",
            files=[("files", ("sample.pdf", b"%PDF-1.4", "application/pdf"))],
        )

    assert response.status_code == 200
    assert response.json() == {
        "uploaded": 1,
        "files": [
            {
                "filename": "sample.pdf",
                "content_type": "application/pdf",
                "size_bytes": 8,
                "parsed": False,
                "text_length": 0,
                "preview": "",
                "chunk_count": 0,
                "stored_chunk_count": 0,
                "graph_chunk_count": 0,
                "entity_count": 0,
                "relation_count": 0,
                "document_id": None,
                "error": "Unable to parse PDF file",
            }
        ],
        "message": "Files received, parsed, chunked, stored in Qdrant, and linked in Neo4j.",
    }


def test_upload_documents_rejects_unsupported_content_type(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "ensure_qdrant_collection", lambda: None)
    monkeypatch.setattr(main_module, "ensure_neo4j_constraints", lambda: None)
    monkeypatch.setattr(main_module, "close_neo4j_driver", lambda: None)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/v1/documents/upload",
            files=[("files", ("image.png", b"pngdata", "image/png"))],
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Unsupported content type: image/png"}


def test_upload_documents_rejects_large_file(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "ensure_qdrant_collection", lambda: None)
    monkeypatch.setattr(main_module, "ensure_neo4j_constraints", lambda: None)
    monkeypatch.setattr(main_module, "close_neo4j_driver", lambda: None)

    large_content = b"a" * ((10 * 1024 * 1024) + 1)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/v1/documents/upload",
            files=[("files", ("big.txt", large_content, "text/plain"))],
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "File too large: big.txt"}