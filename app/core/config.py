from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    app_env: Literal["local", "dev", "staging", "prod"] = Field(default="local", alias="APP_ENV")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")

    streamlit_server_port: int = Field(default=8501, alias="STREAMLIT_SERVER_PORT")
    backend_base_url: str = Field(default="http://localhost:8000", alias="BACKEND_BASE_URL")

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="documents", alias="QDRANT_COLLECTION")
    qdrant_vector_size: int = Field(default=384, alias="QDRANT_VECTOR_SIZE")

    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="neo4jpassword", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    chat_model: str = Field(default="gpt-4.1-mini", alias="CHAT_MODEL")

    llm_provider: Literal["dummy", "openai", "ollama", "local"] = Field(default="local", alias="LLM_PROVIDER")
    local_chat_model: str = Field(default="google/flan-t5-base", alias="LOCAL_CHAT_MODEL")
    local_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="LOCAL_EMBEDDING_MODEL",
    )

    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    top_k_vector: int = Field(default=5, alias="TOP_K_VECTOR")
    top_k_graph: int = Field(default=5, alias="TOP_K_GRAPH")

    @property
    def is_local(self) -> bool:
        return self.app_env == "local"

    def qdrant_auth_kwargs(self) -> dict[str, str]:
        if not self.qdrant_api_key:
            return {}
        return {"api_key": self.qdrant_api_key}

    def neo4j_auth(self) -> tuple[str, str]:
        return self.neo4j_username, self.neo4j_password


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()