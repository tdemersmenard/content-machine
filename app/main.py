from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api import analyses, health, niches, patterns, scripts, videos
from app.config import get_settings
from app.ingest.jobs import Worker


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    worker = Worker() if get_settings().ingest_worker_enabled else None
    if worker:
        worker.start()
    yield
    if worker:
        worker.stop()


app = FastAPI(title="ContentMachine", version=__version__, lifespan=lifespan)
app.include_router(health.router)
app.include_router(videos.router)
app.include_router(analyses.router)
app.include_router(patterns.router)
app.include_router(niches.router)
app.include_router(scripts.router)
