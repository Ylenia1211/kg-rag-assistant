# KG-RAG Assistant

Hybrid Retrieval-Augmented Generation system combining:

- **Qdrant** for vector search
- **Neo4j** for Knowledge Graph storage
- **FastAPI** backend APIs
- **Streamlit** frontend UI
- local / pluggable LLM providers

---

# Features

## Document Ingestion

Upload PDF / TXT / Markdown files.

Pipeline:

1. Parse text
2. Chunk text with overlap
3. Generate embeddings
4. Store vectors in Qdrant
5. Create graph nodes in Neo4j
6. Extract entities and relations

## Hybrid Search

Query flow:

1. Embed user question
2. Semantic retrieval from Qdrant
3. Graph context retrieval from Neo4j
4. Final synthesized answer

## UI

Built with Streamlit:

- Document upload page
- Search / Chat page
- Results inspection

---

# Tech Stack

- Python
- FastAPI
- Streamlit
- Qdrant
- Neo4j
- Docker Compose
- Pytest

---

# Run Locally

## Start databases

```bash
docker compose up -d