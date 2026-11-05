"""Tests du wrapper avec un faux SDK : aucun appel réseau."""

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import LLMCall, LLMCallStatus
from app.llm.client import FALLBACK_BETA, ClaudeClient, LLMCallRecord, db_call_logger
from app.llm.errors import (
    ClaudeInvalidOutputError,
    ClaudeRefusalError,
    ClaudeTruncatedError,
    PromptError,
)

MODEL = "claude-opus-5-5"


class Echo(BaseModel):
    echo: str = Field(min_length=2)


def usage(input_tokens: int = 1000, output_tokens: int = 100, iterations=None, **extra: int):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=extra.get("cache_read", 0),
        cache_creation_input_tokens=extra.get("cache_write", 0),
        iterations=iterations,
    )


def reply(
    text: str = '{"echo": "ok"}',
    stop_reason: str = "end_turn",
    model: str = MODEL,
    message_usage: Any = None,
    stop_details: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop_reason,
        stop_details=stop_details,
        model=model,
        usage=message_usage or usage(),
    )


class FakeStream:
    request_id = "req_test"

    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def get_final_message(self) -> Any:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class FakeSDK:
    """Rejoue une liste de réponses et enregistre les requêtes reçues."""

    def __init__(self, *outcomes: Any) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **params: Any) -> FakeStream:
        self.requests.append(params)
        return FakeStream(self._outcomes.pop(0))


def run(sdk: FakeSDK, records: list[LLMCallRecord] | None = None, **overrides: Any):
    log = records if records is not None else []
    arguments = {
        "prompt": "smoke",
        "version": "v1",
        "schema": Echo,
        "model": MODEL,
        "variables": {"word": "ok"},
    }
    return ClaudeClient(sdk, call_logger=log.append).call_json(**(arguments | overrides))


def test_valid_output_is_parsed_and_logged() -> None:
    records: list[LLMCallRecord] = []

    result = run(FakeSDK(reply()), records)

    assert result.data == Echo(echo="ok")
    assert (result.attempts, result.model, result.prompt_version) == (1, MODEL, "smoke/v1")
    assert result.cost_usd == pytest.approx((1000 * 4 + 100 * 20) / 1_000_000)
    [record] = records
    assert record.status is LLMCallStatus.OK
    assert record.prompt_version == "smoke/v1"
    assert record.prompt_hash
    assert record.request_id == "req_test"
    assert (record.input_tokens, record.output_tokens) == (1000, 100)


def test_request_uses_schema_effort_cache_and_fallback() -> None:
    sdk = FakeSDK(reply())

    run(sdk, effort="xhigh")

    [request] = sdk.requests
    assert request["model"] == MODEL
    assert request["output_config"]["effort"] == "xhigh"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "{{" not in request["messages"][0]["content"][-1]["text"]
    assert request["fallbacks"] == "default"
    assert request["betas"] == [FALLBACK_BETA]
    assert "temperature" not in request


def test_fallback_is_not_sent_for_models_that_do_not_support_it() -> None:
    sdk = FakeSDK(reply(model="claude-haiku-5-5"))

    run(sdk, model="claude-haiku-5-5")

    assert "fallbacks" not in sdk.requests[0]
    assert "betas" not in sdk.requests[0]


@pytest.mark.parametrize("bad_output", ["pas du json", '{"echo": "x"}', '{"autre": 1}', ""])
def test_invalid_output_is_retried_once_with_the_error(bad_output: str) -> None:
    first = reply(bad_output)
    sdk = FakeSDK(first, reply())
    records: list[LLMCallRecord] = []

    result = run(sdk, records)

    assert result.data == Echo(echo="ok")
    assert result.attempts == 2
    assert result.cost_usd == pytest.approx(2 * (1000 * 4 + 100 * 20) / 1_000_000)
    assert [r.status for r in records] == [LLMCallStatus.INVALID_OUTPUT, LLMCallStatus.OK]
    assert records[0].raw_output == bad_output

    retry_messages = sdk.requests[1]["messages"]
    assert len(sdk.requests[0]["messages"]) == 1
    assert [m["role"] for m in retry_messages] == ["user", "assistant", "user"]
    assert retry_messages[1]["content"] is first.content
    assert "validation" in retry_messages[2]["content"]


def test_second_invalid_output_raises() -> None:
    sdk = FakeSDK(reply("non"), reply("toujours non"))
    records: list[LLMCallRecord] = []

    with pytest.raises(ClaudeInvalidOutputError) as caught:
        run(sdk, records)

    assert caught.value.raw_output == "toujours non"
    assert len(sdk.requests) == 2
    assert [r.attempt for r in records] == [1, 2]


def test_refusal_raises_without_retry() -> None:
    details = SimpleNamespace(category="cyber", explanation="non")
    sdk = FakeSDK(reply("", stop_reason="refusal", stop_details=details))
    records: list[LLMCallRecord] = []

    with pytest.raises(ClaudeRefusalError) as caught:
        run(sdk, records)

    assert caught.value.category == "cyber"
    assert len(sdk.requests) == 1
    assert records[0].status is LLMCallStatus.REFUSAL


def test_truncated_output_raises_without_retry() -> None:
    sdk = FakeSDK(reply('{"echo": "o', stop_reason="max_tokens"))
    records: list[LLMCallRecord] = []

    with pytest.raises(ClaudeTruncatedError):
        run(sdk, records)

    assert len(sdk.requests) == 1
    assert records[0].status is LLMCallStatus.TRUNCATED
    assert records[0].raw_output == '{"echo": "o'


def test_api_error_is_logged_and_reraised() -> None:
    error = anthropic.APIConnectionError(request=None, message="réseau coupé")
    records: list[LLMCallRecord] = []

    with pytest.raises(anthropic.APIConnectionError):
        run(FakeSDK(error), records)

    [record] = records
    assert record.status is LLMCallStatus.ERROR
    assert "réseau coupé" in record.error
    assert record.cost_usd is None


def test_fallback_cost_uses_each_served_model() -> None:
    iterations = [
        SimpleNamespace(
            type="message",
            model="claude-opus-5-5",
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        SimpleNamespace(
            type="fallback_message",
            model="claude-opus-4-8",
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    ]
    served = reply(model="claude-opus-4-8", message_usage=usage(iterations=iterations))
    records: list[LLMCallRecord] = []

    result = run(FakeSDK(served), records)

    assert result.model == "claude-opus-4-8"
    assert result.cost_usd == pytest.approx(4 + 5)
    assert records[0].input_tokens == 2_000_000
    assert records[0].model_requested == MODEL
    assert records[0].model_served == "claude-opus-4-8"


def test_unknown_model_keeps_working_without_cost() -> None:
    result = run(FakeSDK(reply(model="modele-futur")), model="modele-futur")

    assert result.data == Echo(echo="ok")
    assert result.cost_usd is None


def test_images_are_sent_before_the_text(tmp_path: Path) -> None:
    frame = tmp_path / "frame.JPG"
    frame.write_bytes(b"image")
    sdk = FakeSDK(reply())

    run(sdk, images=[frame])

    image, text = sdk.requests[0]["messages"][0]["content"]
    assert image["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(image["source"]["data"]) == b"image"
    assert text["type"] == "text"


def test_unsupported_image_format_is_rejected_before_any_call(tmp_path: Path) -> None:
    frame = tmp_path / "frame.bmp"
    frame.write_bytes(b"image")
    sdk = FakeSDK()

    with pytest.raises(ValueError, match="non supporté"):
        run(sdk, images=[frame])

    assert sdk.requests == []


def test_missing_prompt_variable_fails_before_any_call() -> None:
    sdk = FakeSDK()

    with pytest.raises(PromptError):
        run(sdk, variables={})

    assert sdk.requests == []


def test_logging_failure_does_not_lose_the_result() -> None:
    def broken_logger(_: LLMCallRecord) -> None:
        raise RuntimeError("base verrouillée")

    client = ClaudeClient(FakeSDK(reply()), call_logger=broken_logger)

    result = client.call_json(
        prompt="smoke", version="v1", schema=Echo, model=MODEL, variables={"word": "ok"}
    )

    assert result.data == Echo(echo="ok")


def test_every_attempt_is_written_to_llm_calls_even_on_failure(db: Session) -> None:
    client = ClaudeClient(FakeSDK(reply("non"), reply("non")), call_logger=db_call_logger)

    with pytest.raises(ClaudeInvalidOutputError):
        client.call_json(
            prompt="smoke", version="v1", schema=Echo, model=MODEL, variables={"word": "ok"}
        )

    rows = db.scalars(select(LLMCall).order_by(LLMCall.attempt)).all()
    assert [row.status for row in rows] == [LLMCallStatus.INVALID_OUTPUT] * 2
    assert rows[0].prompt_version == "smoke/v1"
    assert rows[0].cost_usd == pytest.approx((1000 * 4 + 100 * 20) / 1_000_000)
    assert rows[0].raw_output == "non"
