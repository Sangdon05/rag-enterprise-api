from typing import Any, Protocol

import psycopg
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from pydantic import SecretStr

from shared.core.config import settings
from shared.core.logger import get_logger

logger = get_logger(__name__)


def get_llm() -> BaseChatModel:
    return init_chat_model(
        model=settings.LLM_MODEL,
        model_provider=settings.LLM_PROVIDER,
        api_key=settings.API_KEY,
        reasoning_effort=settings.REASONING_EFFORT,
    )


def get_embeddings() -> Embeddings:
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=SecretStr(settings.API_KEY),
    )


def get_vector_store(collection_name: str) -> VectorStore:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection_name,
        connection=settings.DATABASE_URL,
        distance_strategy=DistanceStrategy.COSINE,
        use_jsonb=True,
    )


class DBConnection(Protocol):
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...


def get_raw_db_connection() -> DBConnection:
    raw_dsn = settings.DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
    return psycopg.connect(raw_dsn)
