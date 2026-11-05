"""Endpoint de santé. N'appelle jamais Claude."""

import logging
import tempfile
from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app import __version__
from app.config import get_settings
from app.db.session import get_engine

logger = logging.getLogger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    version: str
    database: bool
    data_dir_writable: bool
    anthropic_key_configured: bool


def _database_ok() -> bool:
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        logger.exception("Base de données inaccessible")
        return False
    return True


def _data_dir_writable() -> bool:
    try:
        with tempfile.TemporaryFile(dir=get_settings().data_dir):
            return True
    except OSError:
        return False


@router.get("/health")
def health(response: Response) -> HealthResponse:
    database = _database_ok()
    writable = _data_dir_writable()

    if not database:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="unavailable" if not database else "ok" if writable else "degraded",
        version=__version__,
        database=database,
        data_dir_writable=writable,
        anthropic_key_configured=get_settings().anthropic_api_key is not None,
    )
