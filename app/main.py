from fastapi import FastAPI

from app import __version__
from app.api import health

app = FastAPI(title="ContentMachine", version=__version__)
app.include_router(health.router)
