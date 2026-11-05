from sqlalchemy import inspect, text

from alembic import command
from app.config import get_settings
from app.db.base import Base
from app.db.models import LLMCall  # noqa: F401  (enregistre les tables)
from app.db.session import get_engine
from tests.conftest import alembic_config


def test_upgrade_creates_every_table_then_downgrade_removes_them() -> None:
    config = alembic_config(get_settings().database_url)

    command.upgrade(config, "head")
    tables = set(inspect(get_engine()).get_table_names())
    assert set(Base.metadata.tables) <= tables

    command.downgrade(config, "base")
    assert set(inspect(get_engine()).get_table_names()) <= {"alembic_version"}


def test_ingestion_migration_keeps_existing_videos() -> None:
    config = alembic_config(get_settings().database_url)
    command.upgrade(config, "0001")
    with get_engine().begin() as connection:
        connection.execute(
            text(
                "INSERT INTO videos (is_own_content, created_at) VALUES (0, '2026-01-01 00:00:00')"
            )
        )

    command.upgrade(config, "head")

    with get_engine().connect() as connection:
        assert connection.execute(text("SELECT hashtags FROM videos")).scalar_one() == "[]"
    command.downgrade(config, "0001")
    with get_engine().connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM videos")).scalar_one() == 1


def test_models_match_migrations() -> None:
    config = alembic_config(get_settings().database_url)
    command.upgrade(config, "head")

    command.check(config)  # Lève une erreur si les modèles ont divergé des migrations.
