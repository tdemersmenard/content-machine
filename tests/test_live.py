"""Appel réel à l'API : `uv run pytest -m live`. Coûte quelques centimes au plus."""

import pytest
from pydantic import BaseModel

from app.config import Settings
from app.llm.client import SDK_MAX_RETRIES, ClaudeClient, LLMCallRecord


class Echo(BaseModel):
    echo: str


@pytest.mark.live
def test_real_call_returns_validated_json() -> None:
    import anthropic

    settings = Settings()
    if settings.anthropic_api_key is None:
        pytest.skip("ANTHROPIC_API_KEY absente")

    records: list[LLMCallRecord] = []
    sdk = anthropic.Anthropic(
        api_key=settings.anthropic_api_key.get_secret_value(), max_retries=SDK_MAX_RETRIES
    )

    result = ClaudeClient(sdk, call_logger=records.append).call_json(
        prompt="smoke",
        version="v1",
        schema=Echo,
        model=settings.model_generation,
        variables={"word": "viral"},
        effort="low",
        max_tokens=2000,
    )

    assert result.data.echo == "viral"
    assert records[0].input_tokens > 0
    assert records[0].cost_usd is not None
