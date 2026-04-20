from __future__ import annotations

from typing import Any

import httpx
import streamlit as st

from app.core.config import get_settings


st.set_page_config(
    page_title="Search & Chat",
    page_icon="💬",
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


def search_backend(base_url: str, question: str, limit: int) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/v1/search"
    try:
        response = httpx.post(
            url,
            json={"question": question, "limit": limit},
            timeout=30.0,
        )
        response.raise_for_status()
        return {"ok": True, "payload": response.json(), "error": None}
    except Exception as exc:
        return {"ok": False, "payload": None, "error": str(exc)}


settings = get_settings()
health = fetch_backend_health(settings.backend_base_url)

st.title("💬 Search & Chat")
st.caption("Ricerca ibrida con risposta sintetica LLM, risultati vettoriali, contesto documentale e entità del grafo.")

with st.sidebar:
    st.header("Stato sistema")
    if health["reachable"]:
        st.success(f"Backend: {health['status']}")
        st.json(health["services"])
    else:
        st.error("Backend non raggiungibile")
        if health["error"]:
            st.code(health["error"])

question = st.text_area(
    "Domanda",
    placeholder="Esempio: quali chunk parlano del deployment del servizio X?",
    height=120,
)
limit = st.slider("Numero risultati vettoriali", min_value=1, max_value=10, value=5)

can_search = bool(question.strip()) and health["reachable"]

if st.button("Esegui ricerca", type="primary", disabled=not can_search):
    with st.spinner("Ricerca in corso..."):
        result = search_backend(settings.backend_base_url, question.strip(), limit)

    if not result["ok"]:
        st.error("Ricerca fallita")
        st.code(result["error"] or "Unknown error")
    else:
        payload = result["payload"]
        st.success("Ricerca completata")

        answer = payload.get("answer", "")
        vector_results = payload.get("vector_results", [])
        graph_results = payload.get("graph_results", [])
        entity_results = payload.get("entity_results", [])

        st.subheader("Risposta sintetica")
        if answer:
            st.write(answer)
        else:
            st.info("Nessuna risposta sintetica disponibile.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risultati vettoriali", len(vector_results))
        with col2:
            st.metric("Risultati grafo", len(graph_results))
        with col3:
            st.metric("Entità recuperate", len(entity_results))

        st.subheader("Risultati vettoriali")
        if vector_results:
            for item in vector_results:
                with st.container(border=True):
                    st.write(f"**Point ID**: `{item['id']}`")
                    st.write(f"**Score**: `{item['score']}`")
                    payload_data = item.get("payload", {})
                    st.write(f"**Document ID**: `{payload_data.get('document_id')}`")
                    st.write(f"**Chunk ID**: `{payload_data.get('chunk_id')}`")
                    if payload_data.get("text"):
                        st.code(payload_data["text"])
        else:
            st.info("Nessun risultato vettoriale trovato.")

        st.subheader("Contesto grafo (Chunk)")
        if graph_results:
            for item in graph_results:
                with st.container(border=True):
                    st.write(f"**Document ID**: `{item['document_id']}`")
                    st.write(f"**Chunk ID**: `{item['chunk_id']}`")
                    st.write(f"**Range**: {item['start_char']} - {item['end_char']}")
                    st.code(item["text"])
        else:
            st.info("Nessun contesto grafo disponibile.")

        st.subheader("Entità del grafo")
        if entity_results:
            for item in entity_results:
                with st.container(border=True):
                    st.write(f"**Document ID**: `{item['document_id']}`")
                    st.write(f"**Entity ID**: `{item['entity_id']}`")
                    st.write(f"**Nome**: `{item['entity_name']}`")
                    st.write(f"**Tipo**: `{item['entity_type']}`")
        else:
            st.info("Nessuna entità trovata per i documenti recuperati.")

st.subheader("Passi successivi")
st.markdown(
    """
- migliorare entity extraction e relation extraction con LLM
- fare deduplica/reranking dei chunk
- aggiungere query graph-aware più ricche, non solo `RELATED_TO`
"""
)