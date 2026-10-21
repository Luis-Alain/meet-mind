"""Application entry point: builds the FastAPI app and wires its dependencies."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from meet_mind.agent.factory import AgentFactory, make_agent_factory
from meet_mind.api.routes import router
from meet_mind.config import Settings, get_settings
from meet_mind.repositories.db import Database
from meet_mind.repositories.meetings import MeetingRepository
from meet_mind.services.decider import DeciderClient


def create_app(
    settings: Settings | None = None,
    agent_factory: AgentFactory | None = None,
    decider: DeciderClient | None = None,
) -> FastAPI:
    """Build the app. Passing ``agent_factory`` or ``decider`` replaces the real ones, which is
    how tests run the whole API without calling Bedrock or the Decider server."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Everything the handlers share is created once at startup and released at shutdown.
        resolved = settings or get_settings()
        database = Database(resolved.database_url)
        database.create_all()
        client = decider
        if client is None and resolved.decider_enabled:
            client = DeciderClient(
                resolved.decider_base_url,
                resolved.decider_confidence_threshold,
                timeout=resolved.decider_timeout_seconds,
            )

        app.state.settings = resolved
        app.state.database = database
        app.state.repo = MeetingRepository(database)
        app.state.agent_factory = agent_factory or make_agent_factory(resolved)
        app.state.decider = client
        try:
            yield
        finally:
            if client is not None:
                client.close()
            database.dispose()

    app = FastAPI(
        title="MeetMind",
        version="0.1.0",
        description="Extracts tasks and decisions from meeting transcripts.",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


def run() -> None:
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("meet_mind.main:create_app", factory=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    run()
