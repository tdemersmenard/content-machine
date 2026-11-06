import zlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from alembic.config import Config
from sqlalchemy.orm import Session

from alembic import command
from app.config import get_settings
from app.db.session import get_engine, get_session_factory
from app.llm.client import get_claude
from tests.ingest_helpers import make_video

ROOT = Path(__file__).resolve().parent.parent


def clear_caches() -> None:
    for cached in (get_settings, get_engine, get_session_factory, get_claude):
        cached.cache_clear()


def alembic_config(url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


def distinct_vectors(texts: list[str]) -> np.ndarray:
    """Seuls deux textes identiques se ressemblent : le filtre d'idées reste neutre par défaut."""
    vectors = [
        np.random.default_rng(zlib.crc32(text.encode())).standard_normal(512) for text in texts
    ]
    return np.array([item / np.linalg.norm(item) for item in vectors], dtype=np.float32)


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
    monkeypatch.setenv("AGENT_SCHEDULER_ENABLED", "false")
    monkeypatch.setattr("app.generate.novelty.embed_texts", distinct_vectors)
    clear_caches()
    yield
    get_engine().dispose()
    clear_caches()


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Vidéo de 9 s, 3 plans de couleur, voix synthétique."""
    return make_video(tmp_path_factory.mktemp("media") / "sample.mp4")


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_video(tmp_path_factory.mktemp("media") / "silent.mp4", "yellow", voice=False)


@pytest.fixture
def db() -> Iterator[Session]:
    """Session sur une base créée par les migrations, comme en production."""
    command.upgrade(alembic_config(get_settings().database_url), "head")
    with get_session_factory()() as session:
        yield session
