# LangGraph Agentic Chatbot

A modern, agentic chatbot built with LangGraph that integrates with Confluence, GraphQL API, and Databricks. The chatbot uses a knowledge base to generate accurate queries and provide comprehensive answers.

## Features

- **Agentic Pattern**: Autonomous agent that can search, generate queries, and execute them
- **Multiple Knowledge Sources**: Integration with Confluence, GraphQL API, and Databricks
- **Smart Query Generation**: Uses schema/metadata to generate accurate GraphQL and SQL queries
- **User Hints**: Support for user-provided hints to focus search and improve results
- **Async Operations**: Built with FastAPI for high-performance async operations
- **State Management**: Persistent state using Postgres and PGVector
- **Streaming Support**: Real-time streaming responses
- **Error Handling**: Robust error handling and retry mechanisms

## Prerequisites

- Python 3.9+
- PostgreSQL 13+ with PGVector extension
- Databricks workspace access
- Confluence API access
- GraphQL API access
- AWS Bedrock access (for LLM)

## Local Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd langgraph-tool-conversational
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   # Database
   DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/chatbot_db
   
   # AWS Bedrock
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=your_region
   
   # Confluence
   CONFLUENCE_API_TOKEN=your_token
   CONFLUENCE_URL=your_confluence_url
   
   # Databricks
   DATABRICKS_HOST=your_databricks_host
   DATABRICKS_TOKEN=your_token
   
   # GraphQL
   GRAPHQL_ENDPOINT=your_graphql_endpoint
   GRAPHQL_TOKEN=your_token
   ```

5. **Initialize the database**:
   ```bash
   python scripts/init_db.py
   ```

6. **Run the application**:
   ```bash
   uvicorn app.main:app --reload
   ```

## Usage

### Chat Endpoints

1. **Start a Chat**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "What are the best practices for data modeling?",
       "conversation_id": "optional-existing-id"
     }'
   ```

2. **Stream Chat**:
   ```bash
   curl -X GET "http://localhost:8000/api/v1/chat/stream?message=What%20are%20the%20best%20practices%20for%20data%20modeling%3F&conversation_id=your-id"
   ```

3. **Get Conversation History**:
   ```bash
   curl -X GET "http://localhost:8000/api/v1/conversations/your-id"
   ```

4. **List Conversations**:
   ```bash
   curl -X GET "http://localhost:8000/api/v1/conversations?page=1&limit=10"
   ```

### Using Hints

The chatbot supports various hints to focus the search and improve results:

1. **Source Hints**:
   ```
   #confluence - Search in Confluence
   #graphql - Search in GraphQL schema
   #databricks - Search in Databricks
   ```

2. **Category Hints**:
   ```
   #api - API documentation
   #schema - Schema information
   #docs - General documentation
   #custom_category - Any custom category
   ```

3. **Metadata Hints**:
   ```
   #type:documentation - Filter by type
   #priority:high - Filter by priority
   #version:2.0 - Filter by version
   ```

Example with hints:
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the best practices for data modeling? #confluence #docs #type:documentation",
    "conversation_id": "optional-existing-id"
  }'
```

### Query Generation

The chatbot automatically:
1. Searches relevant knowledge sources
2. Generates appropriate GraphQL/SQL queries
3. Validates queries against schemas
4. Executes queries and formats results

Example GraphQL query generation:
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Get user profile with their recent orders #graphql",
    "conversation_id": "optional-existing-id"
  }'
```

Example SQL query generation:
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me the top 10 customers by revenue #databricks",
    "conversation_id": "optional-existing-id"
  }'
```

## Development

### Project Structure

```
.
├── app/
│   ├── agents/         # Agent implementation
│   ├── api/           # FastAPI routes
│   ├── core/          # Core types and config
│   ├── db/            # Database models and connection
│   ├── services/      # External service integrations
│   └── main.py        # Application entry point
├── scripts/           # Utility scripts
├── tests/            # Test files
├── pyproject.toml    # Project dependencies
└── README.md         # This file
```

### Running Tests

```bash
pytest
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting

Run code quality checks:
```bash
black .
isort .
flake8
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License] 