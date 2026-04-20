# KG-RAG Assistant

Hybrid Retrieval-Augmented Generation platform combining vector search and Knowledge Graph reasoning.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC244C">
  <img src="https://img.shields.io/badge/Neo4j-Graph_DB-4581C3?logo=neo4j&logoColor=white">
  <img src="https://img.shields.io/badge/Pytest-Tested-0A9EDC?logo=pytest&logoColor=white">
</p>

---

## Overview

KG-RAG Assistant is a hybrid AI system that combines:

- **Qdrant** for semantic vector retrieval
- **Neo4j** for Knowledge Graph storage and graph context
- **FastAPI** for backend APIs
- **Streamlit** for the user interface
- **pluggable LLM providers** for final answer synthesis

This architecture improves retrieval quality by combining **semantic similarity** with **structured graph reasoning**.

---


## Features

### Document Ingestion
- Upload PDF / TXT / Markdown files
- Parse and normalize text
- Chunk text with overlap
- Generate embeddings
- Store chunks in Qdrant
- Create graph nodes in Neo4j
- Extract entities and infer relations

### Hybrid Search
- Embed user questions
- Retrieve semantically relevant chunks from Qdrant
- Recover graph context from Neo4j
- Generate a final synthesized answer

---

## UI Preview

### Search & Chat

<p align="center">
  <img src="assets/search.png" width="95%">
</p>

The interface shows:
- the final synthesized answer
- vector retrieval results from Qdrant
- graph context from Neo4j
- extracted entities linked to the retrieved documents

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
```
## Start backend

```bash
uvicorn app.main:app --reload --port 8000
```

## Start UI

```bash
streamlit run streamlit_app/Home.py --server.port 8501
```

## Create a Environment Variables

```bash
BACKEND_BASE_URL=http://localhost:8003

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4jpassword
```
## Testing
```bash
python3 -m pytest -q
```
---
# Roadmap
-LLM-based entity extraction
-Graph visualization page
-Reranking layer
-Source citations
-Multi-document reasoning