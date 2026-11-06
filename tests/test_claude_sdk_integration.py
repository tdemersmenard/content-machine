"""Le wrapper avec le vrai SDK Anthropic, sur un transport HTTP simulé (aucun réseau).

Vérifie ce que le faux SDK de test_claude_client.py ne peut pas prouver : la requête
réellement sérialisée et le parsing réel de la réponse en streaming.
"""

import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
from anthropic import Anthropic, DefaultHttpxClient
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.chat import chat, start_session
from app.agent.routines import WeeklyLearnings
from app.agent.tools import TOOLS
from app.analysis.service import analyze
from app.db.models import LLMCallStatus, Pattern
from app.ingest.jobs import ingest
from app.ingest.schemas import Word
from app.ingest.signals import find_promise
from app.ingest.vision import analyze_batch
from app.llm.client import FALLBACK_BETA, ClaudeClient, LLMCallRecord
from app.patterns.schemas import MergeDecision
from tests.analysis_helpers import sample_analysis
from tests.ingest_helpers import make_deps
from tests.pattern_helpers import FakeEmbedder
from tests.test_niche import make_niche

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


def sse_tool_use(thinking: str, tool_id: str, name: str, tool_input: dict[str, Any]) -> bytes:
    """Une réponse avec un bloc thinking et un appel d'outil, comme l'API en streaming."""
    usage = {"input_tokens": 50, "output_tokens": 1}
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_tool",
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
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": thinking},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig-1"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_input)},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 30},
        },
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def test_agent_loop_through_the_real_sdk(db: Session) -> None:
    """La vraie requête de la boucle : outils, cache, thinking renvoyé, historique rechargeable."""
    client, requests, records = client_replaying(
        sse_tool_use("Je regarde le rapport.", "toolu_1", "learning_report", {}),
        sse("Rien de mesuré."),
        sse("Toujours rien."),
    )
    niche = make_niche(db)
    agent_session = start_session(db, niche.id)

    result = chat(db, agent_session, "Où j'en suis ?", client)
    chat(db, agent_session, "Et là ?", client)

    assert result.answer == "Rien de mesuré."
    assert [step.name for step in result.steps] == ["learning_report", "learning_report"]

    first, second, third = (json.loads(item.content) for item in requests)
    assert {tool["name"] for tool in first["tools"]} == set(TOOLS)
    assert first["cache_control"] == {"type": "ephemeral"}
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["output_config"] == {"effort": "medium"}
    assert first["tools"][0]["eager_input_streaming"] is True

    assistant = second["messages"][1]["content"]
    assert assistant[0] == {
        "type": "thinking",
        "thinking": "Je regarde le rapport.",
        "signature": "sig-1",
    }
    assert assistant[1]["type"] == "tool_use"
    assert second["messages"][2]["content"][0]["tool_use_id"] == "toolu_1"

    # Le thinking ne survit pas au tour : l'historique en base reste du JSON simple.
    assert [block["type"] for block in third["messages"][1]["content"]] == ["tool_use"]
    assert third["messages"][-1] == {"role": "user", "content": "Et là ?"}

    assert [record.prompt_version for record in records] == ["agent_chat/v1"] * 3
    assert records[0].status is LLMCallStatus.OK
    assert records[0].cost_usd is not None and records[0].cost_usd > 0


def test_tool_changes_show_in_the_prompt_hash() -> None:
    client, _, records = client_replaying(sse("a"), sse("b"))
    tool = {"name": "t", "description": "d", "input_schema": {"type": "object"}}
    messages = [{"role": "user", "content": "x"}]

    for description in ("d", "autre"):
        client.call_tools(
            prompt="agent_chat",
            version="v1",
            model=MODEL,
            messages=messages,
            tools=[{**tool, "description": description}],
        )

    assert records[0].prompt_hash != records[1].prompt_hash


def test_weekly_report_schema_through_the_real_sdk() -> None:
    answer = '{"headline": "h", "learned": [], "open_questions": [], "focus": "f"}'
    client, requests, _ = client_replaying(sse(answer))

    result = client.call_json(
        prompt="weekly_report",
        version="v1",
        schema=WeeklyLearnings,
        model=MODEL,
        variables={
            "week_start": "x",
            "readings": "",
            "post_mortems": "",
            "lessons": "",
            "report": "",
        },
    )

    assert result.data.headline == "h"
    schema = json.loads(requests[0].content)["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False


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


def observation(number: int) -> dict[str, Any]:
    return {
        "image_number": number,
        "on_screen_text": "3 ASTUCES",
        "visual_description": "Une personne face caméra",
        "framing": "close_up",
        "face_present": True,
        "camera_motion": "handheld",
        "graphic_elements": ["sous-titres"],
    }


def test_vision_batch_through_the_real_sdk(tmp_path: Path) -> None:
    images = []
    for index in range(2):
        path = tmp_path / f"f{index}.jpg"
        path.write_bytes(b"jpeg")
        images.append((float(index), path))
    reply = {"observations": [observation(2), observation(1)]}
    client, requests, _ = client_replaying(sse(json.dumps(reply)))

    seen = analyze_batch(client, images, MODEL)

    assert [item.image_number for item in seen] == [1, 2]
    body = json.loads(requests[0].content)
    content = body["messages"][0]["content"]
    assert [block["type"] for block in content] == ["image", "image", "text"]
    assert "Image 2 : 1.00 s" in content[2]["text"]
    assert body["output_config"]["effort"] == "medium"
    schema = json.dumps(body["output_config"]["format"]["schema"])
    assert "screen_record" in schema


def test_incomplete_vision_batch_is_corrected_by_the_retry(tmp_path: Path) -> None:
    path = tmp_path / "f.jpg"
    path.write_bytes(b"jpeg")
    images = [(0.0, path), (1.0, path)]
    client, requests, _ = client_replaying(
        sse(json.dumps({"observations": [observation(1)]})),
        sse(json.dumps({"observations": [observation(1), observation(2)]})),
    )

    assert len(analyze_batch(client, images, MODEL)) == 2
    assert "une observation par image" in json.loads(requests[1].content)["messages"][2]["content"]


def test_promise_through_the_real_sdk() -> None:
    words = [Word(word=w, start=i * 1.0, end=i * 1.0 + 0.5) for i, w in enumerate("a b c".split())]
    reply = {"found": True, "promise": "b c", "start_word": 1, "end_word": 2}
    client, requests, _ = client_replaying(sse(json.dumps(reply)))

    promise = find_promise(client, words, MODEL)

    assert (promise.text, promise.start_s, promise.end_s) == ("b c", 1.0, 2.5)
    assert "[0]a [1]b [2]c" in json.loads(requests[0].content)["messages"][0]["content"][0]["text"]


def test_no_promise_through_the_real_sdk() -> None:
    reply = {"found": False, "promise": None, "start_word": None, "end_word": None}
    client, _, _ = client_replaying(sse(json.dumps(reply)))

    assert find_promise(client, [Word(word="a", start=0, end=1)], MODEL) is None


def test_full_analysis_through_the_real_sdk(db: Session, sample_video: Path) -> None:
    video_id = ingest(str(sample_video), deps=make_deps()).video.id
    draft = sample_analysis()
    verified = {"analysis": draft, "corrections": [], "removals": []}
    candidate = {
        "type": "hook",
        "name": "Promesse chiffrée",
        "description": "Annonce un gain mesurable.",
        "template": "Comment [résultat] en [durée]",
        "when_to_use": "Quand le gain est mesurable.",
        "when_not_to_use": "Quand le gain est incertain.",
        "role": "opening",
        "evidence": [
            {"kind": "quote", "start_s": 2.6, "end_s": 3.45, "excerpt": "double your views"}
        ],
    }
    client, requests, records = client_replaying(
        sse(json.dumps(draft)),
        sse(json.dumps(verified)),
        sse(json.dumps({"candidates": [candidate]})),
    )

    row = analyze(db, video_id, client, embedder=FakeEmbedder())

    assert row.prompt_version == "analysis/v1"
    assert row.model == MODEL
    first = json.loads(requests[0].content)
    assert [block["type"] for block in first["messages"][0]["content"]].count("image") >= 5
    # Trop gros pour la contrainte de l'API : le schéma voyage dans le prompt.
    assert "format" not in first["output_config"]
    assert "psychological_mechanism" in first["messages"][0]["content"][-1]["text"]
    second = json.loads(requests[1].content)
    assert "format" not in second["output_config"]
    assert "corrections" in second["messages"][0]["content"][-1]["text"]
    assert all(block["type"] == "text" for block in second["messages"][0]["content"])
    assert [r.prompt_version for r in records] == [
        "analysis/v1",
        "analysis_verify/v1",
        "pattern_extract/v1",
    ]
    assert row.patterns_extracted_at is not None
    assert db.scalars(select(Pattern)).one().name == "Promesse chiffrée"
    extract_schema = json.loads(requests[2].content)["output_config"]["format"]["schema"]
    assert "retention_device" in json.dumps(extract_schema)


def test_merge_decision_through_the_real_sdk() -> None:
    reply = {"same_mechanism": True, "reasoning": "Même ressort", "merged_description": "Enrichie"}
    client, requests, _ = client_replaying(sse(json.dumps(reply)))

    result = client.call_json(
        prompt="pattern_merge",
        version="v1",
        schema=MergeDecision,
        model=MODEL,
        variables={"existing": '{"name": "A"}', "candidate": '{"name": "B"}'},
    )

    assert result.data.merged_description == "Enrichie"
    text = json.loads(requests[0].content)["messages"][0]["content"][0]["text"]
    assert '{"name": "A"}' in text
    assert '{"name": "B"}' in text


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
