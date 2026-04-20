from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import streamlit as st

from app.core.config import get_settings


st.set_page_config(
    page_title="Document Ingestion",
    page_icon="📄",
    layout="wide",
)


@st.cache_data(ttl=5)
def fetch_backend_health(base_url: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/health"
    try:
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
        payload = response.json()
        return {
            "reachable": True,
            "status": payload.get("status", "unknown"),
            "services": payload.get("services", {}),
            "error": None,
        }
    except Exception as exc:
        return {
            "reachable": False,
            "status": "offline",
            "services": {},
            "error": str(exc),
        }


def build_upload_files(uploaded_files: list[Any]) -> list[tuple[str, tuple[str, bytes, str]]]:
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for uploaded_file in uploaded_files:
        mime_type = uploaded_file.type or "application/octet-stream"
        files.append(
            (
                "files",
                (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    mime_type,
                ),
            )
        )
    return files


def upload_documents(base_url: str, uploaded_files: list[Any]) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/v1/documents/upload"
    files = build_upload_files(uploaded_files)

    try:
        response = httpx.post(url, files=files, timeout=60.0)
        response.raise_for_status()
        payload = response.json()
        return {
            "ok": True,
            "payload": payload,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "payload": None,
            "error": str(exc),
        }


def render_file_result(file_info: dict[str, Any]) -> None:
    title = file_info["filename"]
    if file_info["parsed"]:
        st.success(f"{title} · parsing completato")
    else:
        st.warning(f"{title} · parsing non riuscito")

    meta_col1, meta_col2, meta_col3 = st.columns(3)
    with meta_col1:
        st.metric("Chunk generati", file_info["chunk_count"])
    with meta_col2:
        st.metric("Salvati in Qdrant", file_info["stored_chunk_count"])
    with meta_col3:
        st.metric("Salvati in Neo4j", file_info["graph_chunk_count"])

    meta_col4, meta_col5, meta_col6 = st.columns(3)
    with meta_col4:
        st.metric("Entità estratte", file_info["entity_count"])
    with meta_col5:
        st.metric("Relazioni inferite", file_info["relation_count"])
    with meta_col6:
        st.metric("Caratteri estratti", file_info["text_length"])

    st.write(f"**Document ID**: `{file_info['document_id']}`")
    st.write(f"**Content type**: `{file_info['content_type']}`")
    st.write(f"**Dimensione file**: `{file_info['size_bytes']}` bytes")

    if file_info.get("preview"):
        st.write("**Preview**")
        st.code(file_info["preview"])

    if file_info.get("error"):
        st.write("**Errore**")
        st.code(file_info["error"])


settings = get_settings()
health = fetch_backend_health(settings.backend_base_url)

st.title("📄 Document Ingestion")
st.caption("Caricamento file verso il backend per parsing, chunking, persistenza su Qdrant e costruzione del knowledge graph in Neo4j.")

left, right = st.columns([2, 1])

with right:
    st.subheader("Stato backend")
    if health["reachable"]:
        st.success(f"Backend raggiungibile: {health['status']}")
        st.json(health["services"])
    else:
        st.error("Backend non raggiungibile")
        if health["error"]:
            st.code(health["error"])

with left:
    st.subheader("Upload documenti")
    uploaded_files = st.file_uploader(
        "Seleziona uno o più file",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="I file vengono inviati al backend per parsing, chunking, salvataggio vettoriale e costruzione del grafo.",
    )

    if uploaded_files:
        st.write("### File selezionati")
        for file in uploaded_files:
            suffix = Path(file.name).suffix.lower() or "n/a"
            st.write(f"- `{file.name}` · {suffix} · {file.size} bytes")

    can_upload = bool(uploaded_files) and health["reachable"]

    if st.button("Carica documenti", disabled=not can_upload, type="primary"):
        with st.spinner("Invio file al backend in corso..."):
            result = upload_documents(settings.backend_base_url, uploaded_files)

        if result["ok"]:
            st.success("Upload completato.")
            payload = result["payload"]
            st.write(f"**Messaggio backend**: {payload['message']}")
            st.write(f"**File processati**: {payload['uploaded']}")

            for file_info in payload["files"]:
                with st.container(border=True):
                    render_file_result(file_info)
        else:
            st.error("Upload fallito.")
            st.code(result["error"] or "Unknown error")

st.subheader("Cosa fa questa pipeline")
st.markdown(
    """
- validazione file in ingresso
- estrazione testo
- chunking con overlap
- generazione embeddings
- salvataggio chunk in Qdrant
- creazione nodi `Document`, `Chunk`, `Entity` in Neo4j
- inferenza iniziale di relazioni `RELATED_TO`
"""
)