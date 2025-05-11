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
)
from app.services.bedrock import get_bedrock_llm
from app.services.checkpoint import get_checkpoint_manager, get_tool_executor
from app.services.vector_store import (
    get_confluence_retriever,
    get_databricks_retriever,
    get_graphql_schema_retriever,
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
        """Build the agent graph."""
        # Create graph
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("start", self.start_conversation)
        graph.add_node("agent", self.run_agent)
        graph.add_node("end", self.end_conversation)
        
        # Add edges
        graph.add_edge("start", "agent")
        graph.add_edge("agent", "end")
        graph.add_edge("end", END)
        
        # Compile graph with checkpoint manager
        return graph.compile(checkpointer=get_checkpoint_manager())

    async def start_conversation(self, state: AgentState) -> AgentState:
        """Start a new conversation."""
        logger.info("Starting conversation")
        # Initialize state with defaults
        state.update({
            "messages": [],
            "knowledge_results": {},
            "active_sources": [KnowledgeSource.CONFLUENCE, KnowledgeSource.GRAPHQL, KnowledgeSource.DATABRICKS],
            "source_errors": {},
            "schema_context": {},
            "sql_attempts": [],
            "graphql_attempts": [],
            "retry_count": 0,
            "processing_time": 0.0,
        })
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
        
        for pattern, source in source_patterns.items():
            if re.search(pattern, message.lower()):
                hints.source = source
                break
        
        # Parse all hashtags as potential categories or tags
        tag_pattern = r"#(\w+)"
        all_tags = re.findall(tag_pattern, message.lower())
        
        # Filter out source tags
        source_tags = {tag for tag in source_patterns.keys()}
        filtered_tags = [tag for tag in all_tags if f"#{tag}" not in source_tags]
        
        # Add all remaining tags as categories
        hints.categories.extend(filtered_tags)
        
        # Extract metadata from special format: #key:value
        metadata_pattern = r"#(\w+):([^#\s]+)"
        metadata_matches = re.findall(metadata_pattern, message.lower())
        hints.metadata.update({key: value for key, value in metadata_matches})
        
        return hints

    async def run_agent(self, state: AgentState) -> AgentState:
        """Run the autonomous agent."""
        logger.info("Running autonomous agent")
        start_time = time.time()
        question = state["human_message_content"]
        
        try:
            # Parse user hints
            state["user_hints"] = self._parse_user_hints(question)
            
            # Get conversation history
            history = await self._get_conversation_history(state["conversation_id"])
            state["conversation_history"] = history
            
            # Create system prompt for autonomous agent
            system_prompt = """You are an autonomous agent that helps users with their questions about Confluence, GraphQL API, and Databricks.
You have access to the following tools:
1. search_confluence: Search Confluence documents
2. search_graphql_schema: Search GraphQL schema information
3. search_databricks: Search Databricks schema information
4. generate_sql: Generate SQL queries
5. generate_graphql: Generate GraphQL queries
6. validate_sql: Validate SQL queries
7. validate_graphql: Validate GraphQL queries
8. execute_sql: Execute SQL queries
9. execute_graphql: Execute GraphQL queries

Follow these steps:
1. First, search for relevant context in all available sources (Confluence, GraphQL, Databricks)
2. If the question requires SQL or GraphQL, generate and validate queries
3. Execute queries if needed
4. Provide a comprehensive answer based on all gathered information

If you need to use multiple tools, do so in a logical sequence.
If you encounter errors, try alternative approaches.
Always provide clear explanations of your actions.

For follow-up questions:
1. Consider the conversation history
2. Use previous context when relevant
3. Maintain consistency with previous answers
4. If the question is ambiguous, ask for clarification

Pay attention to user hints in the question (e.g., #confluence, #faq, #api) to focus your search."""

            # Create messages for the agent
            messages = [
                SystemMessage(content=system_prompt),
                *[HumanMessage(content=msg["content"]) for msg in history],
                HumanMessage(content=question),
            ]

            # Run the agent
            response = await self.llm.ainvoke(messages)
            
            # Parse the response to determine actions
            actions = self._parse_agent_response(response.content, state["user_hints"])
            
            # Execute actions in parallel where possible
            await self._execute_actions_parallel(state, actions)

            # Generate final answer
            final_answer = await self._generate_final_answer(state)
            state["final_answer"] = final_answer
            
            # Update conversation history
            state["messages"].append({
                "role": "user",
                "content": question,
            })
            state["messages"].append({
                "role": "assistant",
                "content": final_answer,
            })
            
        except Exception as e:
            logger.error(f"Error in agent execution: {e}")
            state["error"] = str(e)
            state["final_answer"] = "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."
        
        finally:
            # Update processing time
            state["processing_time"] = time.time() - start_time
        
        return state

    async def _execute_actions_parallel(self, state: AgentState, actions: List[Dict[str, Any]]) -> None:
        """Execute actions in parallel where possible."""
        # Group actions by type
        search_actions = [a for a in actions if a["tool"] in ["search_confluence", "search_graphql_schema", "search_databricks"]]
        query_actions = [a for a in actions if a["tool"] in ["generate_sql", "generate_graphql", "validate_sql", "validate_graphql", "execute_sql", "execute_graphql"]]
        
        # Execute search actions in parallel
        if search_actions:
            search_tasks = []
            for action in search_actions:
                tool_name = action["tool"]
                state["current_tool"] = tool_name
                state["tool_input"] = action.get("args", {})
                
                # Add user hints to search context
                if state["user_hints"]:
                    state["tool_input"]["hints"] = state["user_hints"]
                
                tool = next((t for t in self.tools if t["name"] == tool_name), None)
                if tool:
                    search_tasks.append(self._execute_tool_safely(tool, state))
            
            # Wait for all search tasks to complete
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Process search results
            for tool_name, result in zip([a["tool"] for a in search_actions], search_results):
                if isinstance(result, Exception):
                    state["source_errors"][tool_name] = str(result)
                else:
                    source = self._get_source_from_tool(tool_name)
                    state["knowledge_results"][source] = [
                        SearchResult(
                            source=source,
                            content=item,
                            relevance_score=self._calculate_relevance(item, state["human_message_content"]),
                            category=self._determine_category(item, state["user_hints"]),
                        )
                        for item in result
                    ]
                    
                    # Update schema context if available
                    if source in [KnowledgeSource.GRAPHQL, KnowledgeSource.DATABRICKS]:
                        state["schema_context"][source] = self._extract_schema_context(result)
        
        # Execute query actions sequentially
        for action in query_actions:
            tool_name = action["tool"]
            state["current_tool"] = tool_name
            state["tool_input"] = action.get("args", {})
            
            # Add schema context to query generation
            if tool_name in ["generate_sql", "generate_graphql"]:
                state["tool_input"]["schema_context"] = state["schema_context"]
            
            try:
                tool = next((t for t in self.tools if t["name"] == tool_name), None)
                if tool:
                    result = await tool["function"](**state["tool_input"])
                    state["tool_output"] = result
                    
                    if tool_name == "generate_sql":
                        state["sql_query"] = result.generated_query
                        state["sql_attempts"].append({
                            "query": result.generated_query,
                            "thought": result.thought,
                            "description": result.description,
                            "schema_context": result.schema_context,
                        })
                    elif tool_name == "generate_graphql":
                        state["graphql_query"] = result.generated_query
                        state["graphql_attempts"].append({
                            "query": result.generated_query,
                            "thought": result.thought,
                            "description": result.description,
                            "schema_context": result.schema_context,
                        })
                    elif tool_name == "validate_sql":
                        state["sql_validation"] = result
                        if not result.is_valid:
                            new_sql = await self._retry_sql_generation(state)
                            if new_sql:
                                state["sql_query"] = new_sql.generated_query
                                state["sql_attempts"].append({
                                    "query": new_sql.generated_query,
                                    "thought": new_sql.thought,
                                    "description": new_sql.description,
                                    "schema_context": new_sql.schema_context,
                                })
                    elif tool_name == "validate_graphql":
                        state["graphql_validation"] = result
                        if not result.is_valid:
                            new_graphql = await self._retry_graphql_generation(state)
                            if new_graphql:
                                state["graphql_query"] = new_graphql.generated_query
                                state["graphql_attempts"].append({
                                    "query": new_graphql.generated_query,
                                    "thought": new_graphql.thought,
                                    "description": new_graphql.description,
                                    "schema_context": new_graphql.schema_context,
                                })
                    elif tool_name == "execute_sql":
                        state["sql_result"] = result
                    elif tool_name == "execute_graphql":
                        state["graphql_result"] = result
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                state["tool_error"] = str(e)
                state["retry_count"] += 1
                
                if state["retry_count"] > 3:
                    raise Exception(f"Max retries exceeded for tool {tool_name}")

    def _determine_category(self, content: Dict[str, Any], hints: Optional[UserHint]) -> Optional[str]:
        """Determine the category of search result based on content and hints."""
        if not hints:
            return None
            
        # Check content metadata for category
        if "category" in content:
            return content["category"]
        
        # Check hints for category
        if hints.categories:
            return hints.categories[0]
        
        return None

    def _extract_schema_context(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract schema context from search results."""
        schema_context = {}
        for result in results:
            if "schema" in result:
                schema_context.update(result["schema"])
        return schema_context

    async def search_confluence(self, query: str, hints: Optional[UserHint] = None) -> List[Dict[str, Any]]:
        """Search Confluence documents."""
        logger.info(f"Searching Confluence: {query}")
        # Add category filter if specified in hints
        if hints and hints.categories:
            category_filters = [f"category:{cat}" for cat in hints.categories]
            query = f"{query} {' '.join(category_filters)}"
            
        # Add metadata filters if specified
        if hints and hints.metadata:
            metadata_filters = [f"{key}:{value}" for key, value in hints.metadata.items()]
            query = f"{query} {' '.join(metadata_filters)}"
            
        return await self.confluence_retriever.ainvoke(query)

    async def search_graphql_schema(self, query: str, hints: Optional[UserHint] = None) -> List[Dict[str, Any]]:
        """Search GraphQL schema information."""
        logger.info(f"Searching GraphQL schema: {query}")
        # Add metadata filters if specified
        if hints and hints.metadata:
            metadata_filters = [f"{key}:{value}" for key, value in hints.metadata.items()]
            query = f"{query} {' '.join(metadata_filters)}"
        
        # Search for schema information
        results = await self.graphql_schema_retriever.ainvoke(query)
        
        # Process and structure schema information
        processed_results = []
        for result in results:
            if "schema" in result:
                # Extract type definitions, fields, and relationships
                schema_info = {
                    "type": result.get("type", ""),
                    "fields": result.get("fields", []),
                    "relationships": result.get("relationships", []),
                    "description": result.get("description", ""),
                    "metadata": result.get("metadata", {}),
                }
                processed_results.append({
                    "content": schema_info,
                    "schema": result["schema"],  # Keep full schema for validation
                })
        
        return processed_results

    async def search_databricks(self, query: str, hints: Optional[UserHint] = None) -> List[Dict[str, Any]]:
        """Search Databricks schema."""
        logger.info(f"Searching Databricks: {query}")
        # Add metadata filters if specified
        if hints and hints.metadata:
            metadata_filters = [f"{key}:{value}" for key, value in hints.metadata.items()]
            query = f"{query} {' '.join(metadata_filters)}"
        
        # Search for table metadata
        results = await self.databricks_retriever.ainvoke(query)
        
        # Process and structure table metadata
        processed_results = []
        for result in results:
            if "table_metadata" in result:
                # Extract table structure, columns, and constraints
                table_info = {
                    "table_name": result.get("table_name", ""),
                    "columns": result.get("columns", []),
                    "constraints": result.get("constraints", []),
                    "description": result.get("description", ""),
                    "metadata": result.get("metadata", {}),
                }
                processed_results.append({
                    "content": table_info,
                    "schema": result["table_metadata"],  # Keep full metadata for validation
                })
        
        return processed_results

    async def generate_sql(
        self,
        question: str,
        context: List[Dict[str, Any]] = None,
        schema_context: Optional[Dict[str, Any]] = None,
        retry_prompt: str = None,
    ) -> SQLGenerationOutput:
        """Generate SQL query."""
        logger.info("Generating SQL query")
        
        # Extract table metadata from context
        table_info = {}
        if context:
            for item in context:
                if "schema" in item:
                    table_info.update(item["schema"])
        
        # Create SQL generation prompt with table metadata
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate a SQL query based on the question, context, and table metadata.
Follow these rules:
1. Use only tables and columns defined in the metadata
2. Respect table relationships and constraints
3. Include appropriate JOIN conditions
4. Add necessary WHERE clauses for filtering
5. Use proper aggregation functions
6. Follow SQL best practices and security guidelines"""),
            ("human", f"""Question: {question}
Context: {context or []}
Table Metadata: {table_info or {}}
{retry_prompt or ''}"""),
        ])
        
        # Generate SQL
        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({})
        
        return SQLGenerationOutput(
            thought="Generated SQL query with table metadata validation",
            description="SQL query following table constraints",
            generated_query=response,
            schema_context=table_info,
        )

    async def generate_graphql(
        self,
        question: str,
        context: List[Dict[str, Any]] = None,
        schema_context: Optional[Dict[str, Any]] = None,
        retry_prompt: str = None,
    ) -> GraphQLGenerationOutput:
        """Generate GraphQL query."""
        logger.info("Generating GraphQL query")
        
        # Extract schema information from context
        schema_info = {}
        if context:
            for item in context:
                if "schema" in item:
                    schema_info.update(item["schema"])
        
        # Create GraphQL generation prompt with schema context
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate a GraphQL query based on the question, context, and schema information.
Follow these rules:
1. Use only fields and types defined in the schema
2. Respect relationships between types
3. Include necessary fragments for complex queries
4. Add appropriate variables for dynamic values
5. Include error handling fields
6. Follow GraphQL best practices"""),
            ("human", f"""Question: {question}
Context: {context or []}
Schema Context: {schema_info or {}}
{retry_prompt or ''}"""),
        ])
        
        # Generate GraphQL
        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({})
        
        return GraphQLGenerationOutput(
            thought="Generated GraphQL query with schema validation",
            description="GraphQL query following schema constraints",
            generated_query=response,
            schema_context=schema_info,
        )

    async def validate_sql(self, sql: str, question: str, schema_context: Optional[Dict[str, Any]] = None) -> ValidationOutput:
        """Validate SQL query."""
        logger.info("Validating SQL query")
        
        # Create validation prompt with table metadata
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Validate the SQL query against the table metadata.
Check for:
1. Valid tables and columns
2. Proper JOIN conditions
3. WHERE clause syntax
4. Aggregation functions
5. Subquery structure
6. Query performance"""),
            ("human", f"""Question: {question}
SQL: {sql}
Table Metadata: {schema_context or {}}"""),
        ])
        
        # Validate SQL
        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({})
        
        return ValidationOutput(
            is_valid="valid" in response.lower(),
            explanation=response,
            schema_validation=schema_context,
        )

    async def validate_graphql(self, query: str, question: str, schema_context: Optional[Dict[str, Any]] = None) -> ValidationOutput:
        """Validate GraphQL query."""
        logger.info("Validating GraphQL query")
        
        # Create validation prompt with schema context
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Validate the GraphQL query against the schema.
Check for:
1. Valid types and fields
2. Proper relationships
3. Required arguments
4. Variable definitions
5. Fragment usage
6. Query complexity"""),
            ("human", f"""Question: {question}
GraphQL: {query}
Schema Context: {schema_context or {}}"""),
        ])
        
        # Validate GraphQL
        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({})
        
        return ValidationOutput(
            is_valid="valid" in response.lower(),
            explanation=response,
            schema_validation=schema_context,
        )

    async def execute_sql(self, sql: str) -> str:
        """Execute SQL query."""
        logger.info("Executing SQL query")
        # Implement SQL execution logic
        return "SQL execution result"

    async def execute_graphql(self, query: str) -> str:
        """Execute GraphQL query."""
        logger.info("Executing GraphQL query")
        # Implement GraphQL execution logic
        return "GraphQL execution result"

    async def _retry_sql_generation(self, state: AgentState) -> Optional[SQLGenerationOutput]:
        """Retry SQL generation with improved context."""
        logger.info("Retrying SQL generation")
        
        # Create retry prompt with previous attempts
        retry_prompt = f"""Previous SQL attempts:
{chr(10).join(f'- {attempt["query"]} (Reason: {attempt["description"]})' for attempt in state["sql_attempts"])}

Please generate a new SQL query that addresses the issues with previous attempts.
Consider the validation feedback: {state["sql_validation"].explanation}"""
        
        # Generate new SQL
        return await self.generate_sql(
            question=state["human_message_content"],
            context=state["knowledge_results"].get(KnowledgeSource.DATABRICKS, []),
            schema_context=state["schema_context"].get(KnowledgeSource.DATABRICKS),
            retry_prompt=retry_prompt,
        )

    async def _retry_graphql_generation(self, state: AgentState) -> Optional[GraphQLGenerationOutput]:
        """Retry GraphQL generation with improved context."""
        logger.info("Retrying GraphQL generation")
        
        # Create retry prompt with previous attempts
        retry_prompt = f"""Previous GraphQL attempts:
{chr(10).join(f'- {attempt["query"]} (Reason: {attempt["description"]})' for attempt in state["graphql_attempts"])}

Please generate a new GraphQL query that addresses the issues with previous attempts.
Consider the validation feedback: {state["graphql_validation"].explanation}"""
        
        # Generate new GraphQL
        return await self.generate_graphql(
            question=state["human_message_content"],
            context=state["knowledge_results"].get(KnowledgeSource.GRAPHQL, []),
            schema_context=state["schema_context"].get(KnowledgeSource.GRAPHQL),
            retry_prompt=retry_prompt,
        )

    async def _generate_final_answer(self, state: AgentState) -> str:
        """Generate final answer based on state."""
        # Create answer generation prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a comprehensive answer based on the gathered information."),
            ("human", f"""Question: {state['human_message_content']}
Conversation History: {state.get('conversation_history', [])}
Knowledge Results:
{self._format_knowledge_results(state['knowledge_results'])}
SQL Query: {state.get('sql_query')}
SQL Result: {state.get('sql_result')}
GraphQL Query: {state.get('graphql_query')}
GraphQL Result: {state.get('graphql_result')}
Previous SQL Attempts: {state.get('sql_attempts', [])}
Previous GraphQL Attempts: {state.get('graphql_attempts', [])}
User Hints: {state.get('user_hints')}"""),
        ])
        
        # Generate answer
        chain = prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({})
        
        return response

    def _format_knowledge_results(self, results: Dict[KnowledgeSource, List[SearchResult]]) -> str:
        """Format knowledge results for prompt."""
        formatted = []
        for source, items in results.items():
            formatted.append(f"{source.value.upper()}:")
            for item in items:
                category_str = f" [{item.category}]" if item.category else ""
                formatted.append(f"- {item.content} (Relevance: {item.relevance_score}){category_str}")
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