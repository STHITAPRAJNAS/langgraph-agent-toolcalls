from functools import lru_cache
from typing import List

from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.retrievers import BaseRetriever

from app.core.config import settings
from app.services.bedrock import get_bedrock_embeddings


@lru_cache()
def get_confluence_retriever() -> BaseRetriever:
    """Get Confluence retriever."""
    embeddings = get_bedrock_embeddings()
    
    # Create PGVector store for Confluence
    vector_store = PGVector(
        connection_string=settings.DATABASE_URL,
        embedding_function=embeddings,
        collection_name=f"{settings.VECTOR_TABLE_PREFIX}confluence",
    )
    
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )


@lru_cache()
def get_databricks_retriever() -> BaseRetriever:
    """Get Databricks retriever."""
    embeddings = get_bedrock_embeddings()
    
    # Create PGVector store for Databricks
    vector_store = PGVector(
        connection_string=settings.DATABASE_URL,
        embedding_function=embeddings,
        collection_name=f"{settings.VECTOR_TABLE_PREFIX}databricks",
    )
    
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    ) 