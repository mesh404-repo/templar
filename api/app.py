"""FastAPI application for templar-tournament."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from tournament.config import get_config
from tournament.storage.database import get_database

from .endpoints import leaderboard, submissions, stats

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

    # Add CORS middleware to allow Vercel frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://bittensor-templar-git-feat-competitions-tplr.vercel.app",
            "https://templar.tplr.ai",  # Production domain
            "http://localhost:3000",  # Local development
            "http://localhost:8000",  # Our dashboard
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_origin_regex=r"https://.*\.vercel\.app",  # All Vercel preview URLs
    )

    # Include routers
    app.include_router(leaderboard.router)
    app.include_router(submissions.router)
    app.include_router(stats.router)

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}

    # Mount static files for dashboard
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


app = create_app()


def main():
    """Run the API server."""
    config = get_config()
    uvicorn.run(
        "api.app:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.debug,
    )


if __name__ == "__main__":
    main()
