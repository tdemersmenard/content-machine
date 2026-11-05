import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.api import health
from app.config import get_settings
from app.main import app

client = TestClient(app)


def test_health_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "0.1.0",
        "database": True,
        "data_dir_writable": True,
        "anthropic_key_configured": False,
    }


def test_health_reports_configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    get_settings.cache_clear()

    assert client.get("/health").json()["anthropic_key_configured"] is True


def test_health_returns_503_when_database_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenEngine:
        def connect(self):
            raise OperationalError("SELECT 1", {}, Exception("base indisponible"))

    monkeypatch.setattr(health, "get_engine", lambda: BrokenEngine())

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert response.json()["database"] is False


def test_health_degraded_when_data_dir_not_writable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "_data_dir_writable", lambda: False)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
