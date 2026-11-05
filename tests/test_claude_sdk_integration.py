"""Le wrapper avec le vrai SDK Anthropic, sur un transport HTTP simulé (aucun réseau).

Vérifie ce que le faux SDK de test_claude_client.py ne peut pas prouver : la requête
réellement sérialisée et le parsing réel de la réponse en streaming.
"""

import json
from typing import Any

import anthropic
import httpx2
from anthropic import Anthropic, DefaultHttpxClient
from pydantic import BaseModel

from app.llm.client import FALLBACK_BETA, ClaudeClient, LLMCallRecord

MODEL = "claude-opus-5-5"


class Echo(BaseModel):
    echo: str


def sse(text: str, usage_extra: dict[str, Any] | None = None) -> bytes:
    usage = {
        "input_tokens": 120,
        "output_tokens": 1,
        "cache_read_input_tokens": 30,
        "cache_creation_input_tokens": 10,
    }
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage,
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 40, **(usage_extra or {})},
        },
        {"type": "message_stop"},
    ]
    lines = [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events]
    return "".join(lines).encode("utf-8")


def client_replaying(
    *bodies: bytes,
) -> tuple[ClaudeClient, list[httpx2.Request], list[LLMCallRecord]]:
    queue = list(bodies)
    requests: list[httpx2.Request] = []
    records: list[LLMCallRecord] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream", "request-id": "req_real"},
            content=queue.pop(0),
        )

    sdk = Anthropic(
        api_key="sk-test",
        max_retries=0,
        http_client=DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return ClaudeClient(sdk, call_logger=records.append), requests, records


def call(client: ClaudeClient):
    return client.call_json(
        prompt="smoke", version="v1", schema=Echo, model=MODEL, variables={"word": "viral"}
    )


def test_request_and_response_through_the_real_sdk() -> None:
    client, requests, records = client_replaying(sse('{"echo": "viral"}'))

    result = call(client)

    assert result.data == Echo(echo="viral")
    [request] = requests
    assert request.url.path == "/v1/messages"
    assert FALLBACK_BETA in request.headers["anthropic-beta"]
    body = json.loads(request.content)
    assert body["stream"] is True
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"]["schema"]["required"] == ["echo"]
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "viral" in body["messages"][0]["content"][0]["text"]

    [record] = records
    assert record.request_id == "req_real"
    assert record.model_served == MODEL
    assert record.input_tokens == 120
    assert record.output_tokens == 40
    assert record.cache_read_tokens == 30
    assert record.cache_write_tokens == 10
    assert record.cost_usd is not None and record.cost_usd > 0


def test_retry_echoes_the_real_content_blocks() -> None:
    client, requests, _ = client_replaying(sse("pas du json"), sse('{"echo": "viral"}'))

    result = call(client)

    assert result.attempts == 2
    retry = json.loads(requests[1].content)["messages"]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    assert retry[1]["content"] == [{"type": "text", "text": "pas du json"}]


def test_usage_iterations_are_read_from_the_real_response() -> None:
    iterations = [
        {
            "type": "message",
            "model": MODEL,
            "input_tokens": 1_000_000,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        {
            "type": "fallback_message",
            "model": "claude-opus-4-8",
            "input_tokens": 1_000_000,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    ]
    client, _, records = client_replaying(
        sse('{"echo": "viral"}', usage_extra={"iterations": iterations})
    )

    result = call(client)

    assert result.cost_usd == 9.0
    assert records[0].input_tokens == 2_000_000


def test_http_error_is_logged_and_keeps_the_sdk_type() -> None:
    records: list[LLMCallRecord] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "x"}}
        )

    sdk = Anthropic(
        api_key="sk-test",
        max_retries=0,
        http_client=DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )

    try:
        ClaudeClient(sdk, call_logger=records.append).call_json(
            prompt="smoke", version="v1", schema=Echo, model=MODEL, variables={"word": "x"}
        )
    except anthropic.BadRequestError:
        pass
    else:
        raise AssertionError("BadRequestError attendue")

    assert records[0].status == "error"
    assert "BadRequestError" in records[0].error
