from __future__ import annotations

import importlib


def load_module():
    return importlib.import_module("streamlit_app.pages.1_Document_Ingestion")


class FakeUploadedFile:
    def __init__(self, name: str, content: bytes, mime_type: str = "text/plain"):
        self.name = name
        self._content = content
        self.type = mime_type
        self.size = len(content)

    def getvalue(self) -> bytes:
        return self._content


class FakeResponse:
    def __init__(self, payload: dict, should_raise: bool = False):
        self._payload = payload
        self._should_raise = should_raise

    def raise_for_status(self) -> None:
        if self._should_raise:
            raise RuntimeError("http error")

    def json(self) -> dict:
        return self._payload


def test_build_upload_files_creates_multipart_payload():
    module = load_module()

    files = [
        FakeUploadedFile("a.txt", b"hello", "text/plain"),
        FakeUploadedFile("b.pdf", b"%PDF", "application/pdf"),
    ]

    payload = module.build_upload_files(files)

    assert payload == [
        ("files", ("a.txt", b"hello", "text/plain")),
        ("files", ("b.pdf", b"%PDF", "application/pdf")),
    ]


def test_upload_documents_success(monkeypatch):
    module = load_module()

    def fake_post(url: str, files, timeout: float):
        assert url == "http://localhost:8000/api/v1/documents/upload"
        assert timeout == 60.0
        assert len(files) == 1
        return FakeResponse(
            {
                "uploaded": 1,
                "files": [
                    {
                        "filename": "a.txt",
                        "content_type": "text/plain",
                        "size_bytes": 5,
                        "parsed": True,
                        "text_length": 5,
                        "preview": "hello",
                        "chunk_count": 1,
                        "stored_chunk_count": 1,
                        "graph_chunk_count": 1,
                        "entity_count": 2,
                        "relation_count": 1,
                        "document_id": "doc-1",
                        "error": None,
                    }
                ],
                "message": "Files received, parsed, chunked, stored in Qdrant, and linked in Neo4j.",
            }
        )

    monkeypatch.setattr(module.httpx, "post", fake_post)

    result = module.upload_documents(
        "http://localhost:8000",
        [FakeUploadedFile("a.txt", b"hello")],
    )

    assert result == {
        "ok": True,
        "payload": {
            "uploaded": 1,
            "files": [
                {
                    "filename": "a.txt",
                    "content_type": "text/plain",
                    "size_bytes": 5,
                    "parsed": True,
                    "text_length": 5,
                    "preview": "hello",
                    "chunk_count": 1,
                    "stored_chunk_count": 1,
                    "graph_chunk_count": 1,
                    "entity_count": 2,
                    "relation_count": 1,
                    "document_id": "doc-1",
                    "error": None,
                }
            ],
            "message": "Files received, parsed, chunked, stored in Qdrant, and linked in Neo4j.",
        },
        "error": None,
    }


def test_upload_documents_failure(monkeypatch):
    module = load_module()

    def fake_post(url: str, files, timeout: float):
        raise RuntimeError("connection error")

    monkeypatch.setattr(module.httpx, "post", fake_post)

    result = module.upload_documents(
        "http://localhost:8000",
        [FakeUploadedFile("a.txt", b"hello")],
    )

    assert result["ok"] is False
    assert result["payload"] is None
    assert "connection error" in result["error"]