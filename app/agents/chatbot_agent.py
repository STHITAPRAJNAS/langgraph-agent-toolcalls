from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio
import time
import re
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.core.types import (
    AgentState,
    FinalAnswerOutput,
    GraphQLGenerationOutput,
    KnowledgeCategory,
    KnowledgeSource,
    SearchResult,
    SQLGenerationOutput,
    UserHint,
    ValidationOutput,
    QueryStatus,
    QueryAttempt,
    QueryHistory,
    KnowledgeResult
)
from app.services.bedrock import get_bedrock_llm
from app.services.checkpoint import get_checkpoint_manager, get_tool_executor
from app.services.vector_store import (
    get_confluence_retriever,
    get_databricks_retriever,
    get_graphql_schema_retriever,
    search_confluence,
    search_graphql_schema,
    search_databricks
)


class ChatbotAgent:
    """Agent for handling chat interactions with tool calling."""

    def __init__(self):
        """Initialize the chatbot agent."""
        self.llm = get_bedrock_llm()
        self.confluence_retriever = get_confluence_retriever()
        self.databricks_retriever = get_databricks_retriever()
        self.graphql_schema_retriever = get_graphql_schema_retriever()
        self.tool_executor = get_tool_executor()
        
        # Initialize tools
        self.tools = [
            {
                "name": "search_confluence",
                "description": "Search Confluence documents for relevant information",
                "function": self.search_confluence,
            },
            {
                "name": "search_graphql_schema",
                "description": "Search GraphQL schema information",
                "function": self.search_graphql_schema,
            },
            {
                "name": "search_databricks",
                "description": "Search Databricks schema information",
                "function": self.search_databricks,
            },
            {
                "name": "generate_sql",
                "description": "Generate SQL query for Databricks",
                "function": self.generate_sql,
            },
            {
                "name": "generate_graphql",
                "description": "Generate GraphQL query",
                "function": self.generate_graphql,
            },
            {
                "name": "validate_sql",
                "description": "Validate generated SQL query",
                "function": self.validate_sql,
            },
            {
                "name": "validate_graphql",
                "description": "Validate generated GraphQL query",
                "function": self.validate_graphql,
            },
            {
                "name": "execute_sql",
                "description": "Execute SQL query on Databricks",
                "function": self.execute_sql,
            },
            {
                "name": "execute_graphql",
                "description": "Execute GraphQL query",
                "function": self.execute_graphql,
            },
        ]
        
        # Initialize graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the agent graph with autonomous execution capabilities."""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each step
        workflow.add_node("start", self.start_conversation)
        workflow.add_node("search_knowledge", self.search_knowledge)
        workflow.add_node("execute_query", self.execute_query)
        workflow.add_node("learn_from_result", self.learn_from_result)
        workflow.add_node("generate_answer", self.generate_answer)
        
        # Define edges with conditions
        workflow.add_edge("start", "search_knowledge")
        workflow.add_edge("search_knowledge", "execute_query")
        workflow.add_edge("execute_query", "learn_from_result")
        workflow.add_edge("learn_from_result", "generate_answer")
        workflow.add_edge("learn_from_result", "execute_query", self._should_retry)
        workflow.add_edge("generate_answer", END)
        
        # Set entry point
        workflow.set_entry_point("start")
        
        # Compile with checkpointing
        return workflow.compile(checkpointer=get_checkpoint_manager())

    async def start_conversation(self, state: AgentState) -> AgentState:
        """Initialize conversation with autonomous capabilities."""
        # Parse user hints for autonomous source selection
        if state.user_hints:
            state.execution_context["active_sources"] = state.user_hints.sources
            state.execution_context["priority"] = state.user_hints.priority
            state.execution_context["constraints"] = state.user_hints.constraints
        
        # Initialize query history
        state.query_history = QueryHistory()
        
        return state

    def _parse_user_hints(self, message: str) -> UserHint:
        """Parse user hints from the message."""
        hints = UserHint()
        
        # Parse source hints
        source_patterns = {
            r"#confluence": KnowledgeSource.CONFLUENCE,
            r"#graphql": KnowledgeSource.GRAPHQL,
            r"#databricks": KnowledgeSource.DATABRICKS,
        }
        
        # Extract source hints
        for pattern, source in source_patterns.items():
            if re.search(pattern, message.lower()):
                hints.sources.append(source)
        
        # Extract metadata from special format: #key:value
        metadata_pattern = r"#(\w+):([^#\s]+)"
        metadata_matches = re.findall(metadata_pattern, message.lower())
        hints.metadata.update({key: value for key, value in metadata_matches})
        
        return hints

    async def search_knowledge(self, state: AgentState) -> AgentState:
        """Autonomously search knowledge sources based on context."""
        active_sources = state.execution_context.get("active_sources", [KnowledgeSource.CONFLUENCE])
        
        for source in active_sources:
            try:
                if source == KnowledgeSource.CONFLUENCE:
                    results = await search_confluence(state.messages[-1]["content"])
                elif source == KnowledgeSource.GRAPHQL:
                    results = await search_graphql_schema(state.messages[-1]["content"])
                elif source == KnowledgeSource.DATABRICKS:
                    results = await search_databricks(state.messages[-1]["content"])
                
                # Create knowledge result with execution plan
                knowledge_result = KnowledgeResult(
                    source=source,
                    content=results[0].content if results else "",
                    metadata=results[0].metadata if results else {},
                    relevance_score=results[0].relevance_score if results else 0.0,
                    execution_plan=self._create_execution_plan(source, results[0] if results else None)
                )
                state.knowledge_results.append(knowledge_result)
                
            except Exception as e:
                state.error_state = {"source": source, "error": str(e)}
        
        return state

    async def execute_query(self, state: AgentState) -> AgentState:
        """Autonomously execute queries with learning capabilities."""
        for result in state.knowledge_results:
            if not result.query_history:
                result.query_history = QueryHistory()
            
            # Generate query based on source
            query = await self._generate_query(result)
            
            # Create new attempt
            attempt = QueryAttempt(
                query=query,
                status=QueryStatus.EXECUTING,
                timestamp=datetime.utcnow()
            )
            result.query_history.attempts.append(attempt)
            result.query_history.total_attempts += 1
            
            try:
                # Execute query
                start_time = time.time()
                query_result = await self._execute_query(result.source, query)
                execution_time = time.time() - start_time
                
                # Update attempt
                attempt.status = QueryStatus.SUCCESS
                attempt.result = query_result
                attempt.execution_time = execution_time
                
                # Update best result
                if not result.query_history.best_result or execution_time < result.query_history.attempts[0].execution_time:
                    result.query_history.best_result = query_result
                
            except Exception as e:
                attempt.status = QueryStatus.ERROR
                attempt.error = str(e)
                state.error_state = {"source": result.source, "error": str(e)}
        
        return state

    async def learn_from_result(self, state: AgentState) -> AgentState:
        """Learn from query execution results and improve future attempts."""
        for result in state.knowledge_results:
            if result.query_history and result.query_history.attempts:
                latest_attempt = result.query_history.attempts[-1]
                
                if latest_attempt.status == QueryStatus.ERROR:
                    # Learn from error
                    improvement = await self._learn_from_error(
                        result.source,
                        latest_attempt.query,
                        latest_attempt.error
                    )
                    if improvement:
                        result.query_history.learned_improvements.append(improvement)
                
                elif latest_attempt.status == QueryStatus.SUCCESS:
                    # Learn from success
                    improvement = await self._learn_from_success(
                        result.source,
                        latest_attempt.query,
                        latest_attempt.result
                    )
                    if improvement:
                        result.query_history.learned_improvements.append(improvement)
        
        return state

    def _should_retry(self, state: AgentState) -> bool:
        """Determine if query should be retried based on learning."""
        if state.retry_count >= state.max_retries:
            return False
            
        for result in state.knowledge_results:
            if result.query_history and result.query_history.attempts:
                latest_attempt = result.query_history.attempts[-1]
                if latest_attempt.status == QueryStatus.ERROR:
                    return True
        
        return False

    async def _generate_query(self, result: KnowledgeResult) -> str:
        """Generate query based on source and context."""
        if result.source == KnowledgeSource.GRAPHQL:
            return await self._generate_graphql_query(result)
        elif result.source == KnowledgeSource.DATABRICKS:
            return await self._generate_sql_query(result)
        return ""

    async def _execute_query(self, source: KnowledgeSource, query: str) -> Any:
        """Execute query based on source."""
        if source == KnowledgeSource.GRAPHQL:
            return await self._execute_graphql_query(query)
        elif source == KnowledgeSource.DATABRICKS:
            return await self._execute_sql_query(query)
        return None

    async def _learn_from_error(self, source: KnowledgeSource, query: str, error: str) -> Optional[str]:
        """Learn from query execution error."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing query errors and suggesting improvements."),
            ("human", f"Query: {query}\nError: {error}\nSource: {source}\nWhat specific improvement can be made to fix this error?")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return await chain.ainvoke({})

    async def _learn_from_success(self, source: KnowledgeSource, query: str, result: Any) -> Optional[str]:
        """Learn from successful query execution."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing successful queries and suggesting optimizations."),
            ("human", f"Query: {query}\nResult: {result}\nSource: {source}\nWhat specific optimization can be made to improve this query?")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return await chain.ainvoke({})

    def _create_execution_plan(self, source: KnowledgeSource, result: Optional[SearchResult]) -> Dict[str, Any]:
        """Create execution plan for query."""
        return {
            "source": source,
            "context": result.metadata if result else {},
            "strategy": "autonomous" if source in [KnowledgeSource.GRAPHQL, KnowledgeSource.DATABRICKS] else "search",
            "priority": 1
        }

    async def _generate_graphql_query(self, result: KnowledgeResult) -> str:
        """Generate GraphQL query with learning from previous attempts."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at generating GraphQL queries."),
            ("human", f"Schema: {result.content}\nContext: {result.metadata}\nPrevious attempts: {result.query_history.attempts if result.query_history else []}\nGenerate a GraphQL query.")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return await chain.ainvoke({})

    async def _generate_sql_query(self, result: KnowledgeResult) -> str:
        """Generate SQL query with learning from previous attempts."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at generating SQL queries."),
            ("human", f"Schema: {result.content}\nContext: {result.metadata}\nPrevious attempts: {result.query_history.attempts if result.query_history else []}\nGenerate a SQL query.")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return await chain.ainvoke({})

    async def _execute_graphql_query(self, query: str) -> Any:
        """Execute GraphQL query."""
        # Implement GraphQL query execution
        pass

    async def _execute_sql_query(self, query: str) -> Any:
        """Execute SQL query."""
        # Implement SQL query execution
        pass

    async def generate_answer(self, state: AgentState) -> AgentState:
        """Generate final answer with execution summary."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at generating comprehensive answers."),
            ("human", f"Context: {state.knowledge_results}\nQuery history: {state.query_history}\nGenerate a detailed answer.")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        answer = await chain.ainvoke({})
        
        state.final_answer = FinalAnswerOutput(
            answer=answer,
            sources=[SearchResult(
                content=r.content,
                source=r.source,
                metadata=r.metadata,
                relevance_score=r.relevance_score,
                query_history=r.query_history
            ) for r in state.knowledge_results],
            confidence=0.9,  # Calculate based on results
            reasoning="Generated from multiple knowledge sources with autonomous execution",
            query_history=state.query_history,
            execution_summary={
                "total_queries": sum(r.query_history.total_attempts for r in state.knowledge_results if r.query_history),
                "successful_queries": sum(1 for r in state.knowledge_results if r.query_history and r.query_history.attempts and r.query_history.attempts[-1].status == QueryStatus.SUCCESS),
                "learned_improvements": [imp for r in state.knowledge_results if r.query_history for imp in r.query_history.learned_improvements]
            }
        )
        
        return state

    def _format_knowledge_results(self, results: Dict[KnowledgeSource, List[SearchResult]]) -> str:
        """Format knowledge results for prompt."""
        formatted = []
        for source, items in results.items():
            formatted.append(f"{source.value.upper()}:")
            for item in items:
                formatted.append(f"- {item.content} (Relevance: {item.relevance_score})")
        return "\n".join(formatted)

    async def end_conversation(self, state: AgentState) -> AgentState:
        """End the conversation."""
        logger.info("Ending conversation")
        return state

    async def invoke(self, message: str, conversation_id: str) -> Dict[str, Any]:
        """Invoke the agent with a message."""
        # Create initial state
        state = {
            "conversation_id": conversation_id,
            "human_message_content": message,
        }
        
        # Run graph
        result = await self.graph.ainvoke(state)
        return result

    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history from checkpoints."""
        checkpoint_manager = get_checkpoint_manager()
        checkpoints = await checkpoint_manager.list_checkpoints(conversation_id)
        
        history = []
        for checkpoint in checkpoints:
            state = await checkpoint_manager.get_checkpoint(checkpoint)
            if state and "messages" in state:
                history.extend(state["messages"])
        
        return history

    def _parse_agent_response(self, response: str, hints: Optional[UserHint] = None) -> List[Dict[str, Any]]:
        """Parse the agent's response to determine actions."""
        actions = []
        
        # Look for tool usage patterns in the response
        if "search_confluence" in response.lower() or (hints and hints.source == KnowledgeSource.CONFLUENCE):
            actions.append({
                "tool": "search_confluence",
                "args": {"query": response, "hints": hints},
            })
        
        if "search_graphql" in response.lower() or (hints and hints.source == KnowledgeSource.GRAPHQL):
            actions.append({
                "tool": "search_graphql_schema",
                "args": {"query": response, "hints": hints},
            })
        
        if "search_databricks" in response.lower() or (hints and hints.source == KnowledgeSource.DATABRICKS):
            actions.append({
                "tool": "search_databricks",
                "args": {"query": response, "hints": hints},
            })
        
        if "generate_sql" in response.lower():
            actions.append({
                "tool": "generate_sql",
                "args": {"question": response},
            })
        
        if "generate_graphql" in response.lower():
            actions.append({
                "tool": "generate_graphql",
                "args": {"question": response},
            })
        
        return actions 