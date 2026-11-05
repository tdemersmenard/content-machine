from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy.orm import Session

from alembic import command
from app.config import get_settings
from app.db.session import get_engine, get_session_factory
from app.llm.client import get_claude

ROOT = Path(__file__).resolve().parent.parent


def clear_caches() -> None:
    for cached in (get_settings, get_engine, get_session_factory, get_claude):
        cached.cache_clear()


def alembic_config(url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


@pytest.fixture(autouse=True)
def isolated_env(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Chaque test a son propre DATA_DIR et aucune clé API (sauf les tests live)."""
    if request.node.get_closest_marker("live"):
        yield
        return

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    clear_caches()
    yield
    get_engine().dispose()
    clear_caches()


@pytest.fixture
def db() -> Iterator[Session]:
    """Session sur une base créée par les migrations, comme en production."""
    command.upgrade(alembic_config(get_settings().database_url), "head")
    with get_session_factory()() as session:
        yield session
