from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from app import __version__
from app.agent.scheduler import build_scheduler
from app.api import agent, analyses, health, learning, niches, patterns, scripts, videos
from app.config import get_settings
from app.ingest.jobs import Worker
from app.web import routes as web
from app.web.security import protect

STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "static"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    worker = Worker() if settings.ingest_worker_enabled else None
    scheduler = build_scheduler() if settings.agent_scheduler_enabled else None
    if worker:
        worker.start()
    if scheduler:
        scheduler.start()
    yield
    if scheduler:
        scheduler.shutdown(wait=False)
    if worker:
        worker.stop()


app = FastAPI(title="ContentMachine", version=__version__, lifespan=lifespan)
app.add_middleware(BaseHTTPMiddleware, dispatch=protect)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(health.router)
app.include_router(videos.router)
app.include_router(analyses.router)
app.include_router(patterns.router)
app.include_router(niches.router)
app.include_router(scripts.router)
app.include_router(learning.router)
app.include_router(agent.router)
app.include_router(web.router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/ui/inbox", status_code=303)
