from typing import Any, Dict, List, Optional, TypedDict, Union
from enum import Enum
from pydantic import BaseModel, Field


class KnowledgeSource(str, Enum):
    """Enum for different knowledge sources."""
    CONFLUENCE = "confluence"
    GRAPHQL = "graphql"
    DATABRICKS = "databricks"


class KnowledgeCategory(str, Enum):
    """Enum for different knowledge categories."""
    FAQ = "faq"
    API = "api"
    SCHEMA = "schema"
    METADATA = "metadata"
    DOCUMENTATION = "documentation"
    BEST_PRACTICES = "best_practices"


class UserHint(BaseModel):
    """Model for user hints in questions."""
    source: Optional[KnowledgeSource] = None
    categories: List[str] = Field(default_factory=list)  # Generic categories
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Additional metadata for hints


class SearchResult(BaseModel):
    """Model for search results from any source."""
    source: KnowledgeSource
    content: Dict[str, Any]
    relevance_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    category: Optional[str] = None  # Generic category


class SQLGenerationOutput(BaseModel):
    """Output from SQL generation."""
    thought: str
    description: str
    generated_query: str
    schema_context: Optional[Dict[str, Any]] = None


class GraphQLGenerationOutput(BaseModel):
    """Output from GraphQL query generation."""
    thought: str
    description: str
    generated_query: str
    schema_context: Optional[Dict[str, Any]] = None


class ValidationOutput(BaseModel):
    """Output from query validation."""
    is_valid: bool
    explanation: str
    schema_validation: Optional[Dict[str, Any]] = None


class FinalAnswerOutput(BaseModel):
    """Final answer output."""
    answer: str
    sources: List[SearchResult] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    knowledge_sources: List[KnowledgeSource] = Field(default_factory=list)
    used_hints: Optional[UserHint] = None


class AgentState(TypedDict, total=False):
    """State for the agent."""
    # Core state
    conversation_id: str
    human_message_content: str
    messages: List[Dict[str, Any]]
    user_hints: Optional[UserHint]
    
    # Tool execution state
    current_tool: Optional[str]
    tool_input: Optional[Dict[str, Any]]
    tool_output: Optional[Any]
    tool_error: Optional[str]
    
    # Knowledge state
    knowledge_results: Dict[KnowledgeSource, List[SearchResult]]
    active_sources: List[KnowledgeSource]
    source_errors: Dict[KnowledgeSource, str]
    schema_context: Dict[KnowledgeSource, Dict[str, Any]]
    
    # Query state
    sql_query: Optional[str]
    sql_validation: Optional[ValidationOutput]
    sql_result: Optional[Any]
    sql_attempts: List[Dict[str, Any]]
    
    graphql_query: Optional[str]
    graphql_validation: Optional[ValidationOutput]
    graphql_result: Optional[Any]
    graphql_attempts: List[Dict[str, Any]]
    
    # Final state
    final_answer: Optional[str]
    error: Optional[str]
    retry_count: int
    processing_time: float


class ChatResponse(BaseModel):
    """Response model for chat endpoints."""
    conversation_id: str
    message: str
    conversation_name: Optional[str] = None
    sources: List[SearchResult] = Field(default_factory=list)
    processing_time: float
    used_hints: Optional[UserHint] = None 