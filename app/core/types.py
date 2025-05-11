from typing import Any, Dict, List, Optional, TypedDict, Union
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class KnowledgeSource(str, Enum):
    """Enum for different knowledge sources."""
    CONFLUENCE = "confluence"
    GRAPHQL = "graphql"
    DATABRICKS = "databricks"


class QueryStatus(str, Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    ERROR = "error"
    RETRYING = "retrying"
    FAILED = "failed"


class QueryAttempt(BaseModel):
    query: str
    status: QueryStatus
    error: Optional[str] = None
    result: Optional[Any] = None
    execution_time: Optional[float] = None
    feedback: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QueryHistory(BaseModel):
    attempts: List[QueryAttempt] = Field(default_factory=list)
    total_attempts: int = 0
    best_result: Optional[Any] = None
    learned_improvements: List[str] = Field(default_factory=list)


class KnowledgeResult(BaseModel):
    source: KnowledgeSource
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float
    query_history: Optional[QueryHistory] = None
    execution_plan: Optional[Dict[str, Any]] = None


class UserHint(BaseModel):
    """Model for user hints in questions."""
    sources: List[KnowledgeSource] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: Optional[int] = None
    constraints: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    """Model for search results from any source."""
    content: str
    source: KnowledgeSource
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float
    query_history: Optional[QueryHistory] = None


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
    sources: List[SearchResult]
    confidence: float
    reasoning: str
    query_history: Optional[QueryHistory] = None
    execution_summary: Optional[Dict[str, Any]] = None


class AgentState(BaseModel):
    messages: List[Dict[str, str]] = Field(default_factory=list)
    knowledge_results: List[KnowledgeResult] = Field(default_factory=list)
    current_query: Optional[str] = None
    query_history: Optional[QueryHistory] = None
    user_hints: Optional[UserHint] = None
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    learning_state: Dict[str, Any] = Field(default_factory=dict)
    error_state: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    conversation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatInput(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_hints: Optional[UserHint] = None
    execution_context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoints."""
    response: str
    conversation_id: str
    sources: List[SearchResult]
    confidence: float
    query_history: Optional[QueryHistory] = None
    execution_summary: Optional[Dict[str, Any]] = None 