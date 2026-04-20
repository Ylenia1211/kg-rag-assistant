from __future__ import annotations

from typing import Any

import httpx
import streamlit as st

from app.core.config import get_settings


st.set_page_config(
    page_title="KG RAG Assistant",
    page_icon="🧠",
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


def render_status_card(title: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=title, value=value, help=help_text)


settings = get_settings()
health = fetch_backend_health(settings.backend_base_url)

st.title("🧠 KG + Neo4j + RAG Assistant")
st.caption("Dashboard iniziale per backend FastAPI, Qdrant, Neo4j e interfaccia Streamlit.")

with st.sidebar:
    st.header("Configurazione")
    st.write(f"**Backend**: `{settings.backend_base_url}`")
    st.write(f"**Qdrant collection**: `{settings.qdrant_collection}`")
    st.write(f"**Neo4j database**: `{settings.neo4j_database}`")
    st.write(f"**Embedding model**: `{settings.embedding_model}`")
    st.divider()
    st.info(
        "Prossimi step: upload documenti, ingestion pipeline, retrieval ibrido e chat RAG+KG."
    )

col1, col2, col3 = st.columns(3)

with col1:
    render_status_card(
        "Backend",
        "online" if health["reachable"] else "offline",
        help_text=health.get("error"),
    )

with col2:
    render_status_card(
        "Qdrant",
        "online" if health["services"].get("qdrant") else "offline",
    )

with col3:
    render_status_card(
        "Neo4j",
        "online" if health["services"].get("neo4j") else "offline",
    )

st.subheader("Stato sistema")
if health["reachable"]:
    if health["status"] == "ok":
        st.success("Tutti i servizi principali risultano disponibili.")
    else:
        st.warning("Il backend risponde, ma uno o più servizi risultano degradati.")
else:
    st.error("Il backend non è raggiungibile. Controlla porta, URL e processo uvicorn.")
    if health["error"]:
        st.code(health["error"])

st.subheader("Roadmap operativa")
st.markdown(
    """
1. Verifica backend e servizi.
2. Aggiungi upload documenti.
3. Costruisci pipeline di chunking ed embeddings.
4. Salva chunk in Qdrant ed entità/relazioni in Neo4j.
5. Implementa ricerca ibrida e chat finale.
"""
)

st.subheader("Prossime pagine")
left, right = st.columns(2)

with left:
    st.info("Document Ingestion — caricamento file, parsing, chunking e indicizzazione")

with right:
    st.info("Chat Assistant — domanda, retrieval ibrido, evidenze e risposta finale")