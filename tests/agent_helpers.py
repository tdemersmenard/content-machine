"""Doubles de test pour l'agent."""

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import LLMCall, LLMCallStatus
from app.llm.client import ClaudeResult, ToolTurn
from tests.script_helpers import FakeScriptClaude


@dataclass
class Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


def text_turn(text: str) -> list[Block]:
    return [Block("text", text=text)]


def tool_turn(name: str, tool_input: dict[str, Any] | None = None, say: str = "") -> list[Block]:
    blocks = [Block("text", text=say)] if say else []
    blocks.append(Block("tool_use", id=f"toolu_{name}", name=name, input=tool_input or {}))
    return blocks


class FakeAgentClaude(FakeScriptClaude):
    """`turns` donne les réponses successives de la boucle d'outils."""

    def __init__(self, turns: list[list[Block]] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.turns = list(turns or [])
        self.tool_calls: list[dict[str, Any]] = []

    def call_tools(self, *, prompt: str, version: str, model: str, messages, tools, **_: Any):
        self.tool_calls.append(
            {"prompt": prompt, "version": version, "messages": list(messages), "tools": tools}
        )
        content = self.turns.pop(0)
        if isinstance(content, Exception):
            raise content
        stop = "tool_use" if any(item.type == "tool_use" for item in content) else "end_turn"
        return ToolTurn(
            content=content,
            stop_reason=stop,
            model=model,
            prompt_version=f"{prompt}/{version}",
            cost_usd=0.01,
        )


class FakeWeeklyClaude(FakeScriptClaude):
    """Ajoute le rapport de la semaine et le post-mortem aux réponses de génération."""

    def __init__(self, learned: list[dict] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.learned = learned or []

    def call_json(self, *, prompt: str, version: str, schema: Any, model: str, **kwargs: Any):
        if prompt == "weekly_report":
            self.calls.append({"prompt": prompt, "version": version, **kwargs})
            data = schema(
                headline="Une semaine de test",
                learned=self.learned,
                open_questions=["Il manque des vidéos"],
                focus="Tourner plus",
            )
        elif prompt == "post_mortem":
            self.calls.append({"prompt": prompt, "version": version, **kwargs})
            data = schema(
                summary="Diagnostic",
                causes=[
                    {
                        "category": "hook",
                        "weight_percent": 100,
                        "explanation": "Le hook",
                        "evidence": "retention_3s",
                    }
                ],
                script_vs_actual="Non ingérée",
                keep=["Garder"],
                change=["Changer"],
                prediction_review="Trop optimiste",
            )
        else:
            return super().call_json(
                prompt=prompt, version=version, schema=schema, model=model, **kwargs
            )
        return ClaudeResult(
            data=data, model=model, prompt_version=f"{prompt}/{version}", attempts=1, cost_usd=0.01
        )


def spend(db: Session, tokens: int, cache_read: int = 0, created_at=None) -> None:
    """Simule une consommation dans `llm_calls`."""
    row = LLMCall(
        prompt_version="test/v1",
        prompt_hash="0",
        model_requested="m",
        attempt=1,
        status=LLMCallStatus.OK,
        latency_ms=1,
        input_tokens=tokens,
        cache_read_tokens=cache_read,
        cost_usd=0.5,
    )
    if created_at is not None:
        row.created_at = created_at
    db.add(row)
    db.commit()
