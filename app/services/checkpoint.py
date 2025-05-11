from typing import Optional

from langgraph.checkpoint import AsyncPostgresSaver
from langgraph.prebuilt import ToolExecutor

from app.core.config import settings
from app.agents.chatbot_agent import ChatbotAgent

# Global checkpoint manager
checkpoint_manager: Optional[AsyncPostgresSaver] = None
tool_executor: Optional[ToolExecutor] = None


async def init_checkpoint_manager() -> None:
    """Initialize the checkpoint manager."""
    global checkpoint_manager, tool_executor
    
    # Initialize checkpoint manager
    checkpoint_manager = AsyncPostgresSaver(
        connection_string=settings.DATABASE_URL,
        table_name="langgraph_checkpoints",
    )
    
    # Initialize tool executor
    chatbot = ChatbotAgent()
    tool_executor = ToolExecutor(chatbot.tools)


def get_checkpoint_manager() -> AsyncPostgresSaver:
    """Get the checkpoint manager instance."""
    if checkpoint_manager is None:
        raise RuntimeError("Checkpoint manager not initialized")
    return checkpoint_manager


def get_tool_executor() -> ToolExecutor:
    """Get the tool executor instance."""
    if tool_executor is None:
        raise RuntimeError("Tool executor not initialized")
    return tool_executor 