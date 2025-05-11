from typing import AsyncGenerator, List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.chatbot_agent import ChatbotAgent
from app.core.types import ChatResponse
from app.db.connection import get_session
from app.db.models import ConversationMetadata, ConversationState
from app.services.checkpoint import get_checkpoint_manager

router = APIRouter()
chatbot = ChatbotAgent()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    message: str,
    conversation_id: str = None,
    session: AsyncSession = Depends(get_session),
) -> ChatResponse:
    """Chat endpoint."""
    if not conversation_id:
        conversation_id = str(uuid4())
    
    # Get or create conversation metadata
    metadata = await session.get(ConversationMetadata, conversation_id)
    if not metadata:
        metadata = ConversationMetadata(
            id=str(uuid4()),
            conversation_id=conversation_id,
        )
        session.add(metadata)
        await session.commit()
    
    # Invoke chatbot
    result = await chatbot.invoke(message, conversation_id)
    
    # Save state
    state = ConversationState(
        id=str(uuid4()),
        conversation_id=conversation_id,
        state=result,
    )
    session.add(state)
    await session.commit()
    
    return ChatResponse(
        conversation_id=conversation_id,
        message=result["final_answer"],
        conversation_name=metadata.name,
    )


@router.get("/chat/stream")
async def chat_stream(
    message: str,
    conversation_id: str = None,
    session: AsyncSession = Depends(get_session),
) -> EventSourceResponse:
    """Streaming chat endpoint."""
    if not conversation_id:
        conversation_id = str(uuid4())
    
    # Get or create conversation metadata
    metadata = await session.get(ConversationMetadata, conversation_id)
    if not metadata:
        metadata = ConversationMetadata(
            id=str(uuid4()),
            conversation_id=conversation_id,
        )
        session.add(metadata)
        await session.commit()
    
    async def event_generator() -> AsyncGenerator[dict, None]:
        """Generate SSE events."""
        try:
            # Invoke chatbot with streaming
            async for chunk in chatbot.graph.astream(
                {
                    "messages": [],
                    "conversation_id": conversation_id,
                    "human_message_content": message,
                    "intermediate_steps": [],
                    "intermediate_answers": [],
                }
            ):
                if chunk.get("final_answer"):
                    yield {
                        "event": "message",
                        "data": chunk["final_answer"],
                    }
        except Exception as e:
            yield {
                "event": "error",
                "data": str(e),
            }
    
    return EventSourceResponse(event_generator())


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get conversation history."""
    # Get conversation metadata
    metadata = await session.get(ConversationMetadata, conversation_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Get conversation history from checkpoints
    history = await chatbot.get_conversation_history(conversation_id)
    
    return {
        "conversation_id": conversation_id,
        "name": metadata.name,
        "messages": history,
    }


@router.get("/conversations")
async def list_conversations(
    session: AsyncSession = Depends(get_session),
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> List[dict]:
    """List all conversations."""
    # Get conversations from metadata
    query = session.query(ConversationMetadata).offset(offset).limit(limit)
    conversations = await session.execute(query)
    results = conversations.scalars().all()
    
    return [
        {
            "conversation_id": conv.conversation_id,
            "name": conv.name,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
        }
        for conv in results
    ]


@router.get("/conversations/{conversation_id}/checkpoints")
async def get_conversation_checkpoints(
    conversation_id: str,
) -> List[dict]:
    """Get all checkpoints for a conversation."""
    checkpoint_manager = get_checkpoint_manager()
    checkpoints = await checkpoint_manager.list_checkpoints(conversation_id)
    
    return [
        {
            "checkpoint_id": checkpoint,
            "timestamp": checkpoint.split("_")[-1],  # Assuming timestamp in checkpoint ID
        }
        for checkpoint in checkpoints
    ] 