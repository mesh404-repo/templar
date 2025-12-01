"""FastAPI application for templar-tournament."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from tournament.config import get_config
from tournament.storage.database import get_database

from .endpoints import leaderboard, submissions

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting API server...")
    await get_database()  # Initialize database
    yield
    # Shutdown
    logger.info("Shutting down API server...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Templar Tournament",
        description="Training code efficiency competition on Bittensor",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Include routers
    app.include_router(leaderboard.router)
    app.include_router(submissions.router)

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}

    return app


app = create_app()


def main():
    """Run the API server."""
    import uvicorn

    config = get_config()
    uvicorn.run(
        "api.app:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.debug,
    )


if __name__ == "__main__":
    main()
