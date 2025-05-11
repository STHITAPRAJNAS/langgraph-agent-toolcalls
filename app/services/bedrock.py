from functools import lru_cache

from langchain_community.llms.bedrock import Bedrock
from langchain_community.embeddings.bedrock import BedrockEmbeddings

from app.core.config import settings


@lru_cache()
def get_bedrock_llm() -> Bedrock:
    """Get Bedrock LLM instance."""
    return Bedrock(
        model_id=settings.BEDROCK_MODEL_ID,
        region_name=settings.AWS_REGION,
        credentials_profile_name=None,  # Use environment variables
    )


@lru_cache()
def get_bedrock_embeddings() -> BedrockEmbeddings:
    """Get Bedrock embeddings instance."""
    return BedrockEmbeddings(
        model_id=settings.TITAN_EMBEDDING_MODEL_ID,
        region_name=settings.AWS_REGION,
        credentials_profile_name=None,  # Use environment variables
    ) 