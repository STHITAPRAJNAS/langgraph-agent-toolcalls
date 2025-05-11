from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router as api_router
from app.core.config import settings
from app.db.connection import init_db, close_db
from app.services.checkpoint import init_checkpoint_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI application."""
    # Startup
    logger.info("Starting up Knowledge Chatbot API...")
    await init_db()
    await init_checkpoint_manager()
    yield
    # Shutdown
    logger.info("Shutting down Knowledge Chatbot API...")
    await close_db()


app = FastAPI(
    title="Knowledge Chatbot API",
    description="API for a knowledge chatbot using LangGraph",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    ) 