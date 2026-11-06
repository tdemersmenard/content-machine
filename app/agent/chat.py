"""La boucle d'outils de l'agent.

Boucle manuelle plutôt que le tool runner du SDK : chaque tour doit passer par le budget, le journal
et la journalisation de `llm_calls`.

L'historique gardé en base ne contient ni le thinking ni les blocs bruts du SDK : un thinking n'est
utile que pendant la boucle d'un même tour, où les blocs sont renvoyés tels quels.
"""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import anthropic
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent.budget import TOKENS_CHAT_ROUND, BudgetExceeded, check_budget
from app.agent.journal import logged_action
from app.agent.memory import load_memory
from app.agent.tools import ToolContext, api_tools, run_tool
from app.config import get_settings
from app.db.models import (
    ActionStatus,
    AgentSession,
    Lesson,
    LLMCall,
    Niche,
    Pattern,
    Script,
    Video,
)
from app.ingest.pipeline import IngestDeps
from app.llm.client import ClaudeClient
from app.llm.errors import ClaudeCredentialsError, ClaudeError
from app.llm.prompts import load_prompt
from app.patterns.embeddings import Embedder

logger = logging.getLogger(__name__)

PROMPT = ("agent_chat", "v1")
MAX_LOGGED_MESSAGE_CHARS = 500

_CITATION_RE = re.compile(r"\[(vidéo|video|pattern|leçon|lecon|script)\s+#?(\d+)\]", re.IGNORECASE)
_CITATION_MODELS = {"video": Video, "pattern": Pattern, "lesson": Lesson, "script": Script}
_CITATION_KINDS = {
    "vidéo": "video",
    "video": "video",
    "pattern": "pattern",
    "leçon": "lesson",
    "lecon": "lesson",
    "script": "script",
}


class AgentError(Exception):
    """La conversation ne peut pas démarrer."""


@dataclass(frozen=True)
class Step:
    """Ce que l'agent montre pendant qu'il travaille."""

    kind: str  # « text », « tool_start » ou « tool_end ».
    name: str = ""
    detail: str = ""
    is_error: bool = False


@dataclass
class ChatResult:
    session_id: int
    answer: str
    steps: list[Step] = field(default_factory=list)
    evidence: dict[str, list[int]] = field(default_factory=dict)
    unverified: list[str] = field(default_factory=list)  # Références qui n'existent pas en base.
    interrupted: str | None = None  # Raison de l'arrêt, si le tour n'a pas pu finir.
    tokens: int = 0
    cost_usd: float = 0.0


def check_citations(session: Session, text: str) -> tuple[dict[str, list[int]], list[str]]:
    """Le code a le dernier mot : une référence à un id absent de la base est signalée."""
    cited: dict[str, set[int]] = {kind: set() for kind in _CITATION_MODELS}
    for label, value in _CITATION_RE.findall(text):
        cited[_CITATION_KINDS[label.lower()]].add(int(value))

    evidence: dict[str, list[int]] = {}
    unverified: list[str] = []
    for kind, ids in cited.items():
        model = _CITATION_MODELS[kind]
        known = set(session.scalars(select(model.id).where(model.id.in_(ids))))
        if known:
            evidence[kind] = sorted(known)
        unverified += [f"{kind} {item}" for item in sorted(ids - known)]
    return evidence, unverified


def start_session(session: Session, niche_id: int) -> AgentSession:
    niche = session.get(Niche, niche_id)
    if niche is None or niche.profile is None:
        raise AgentError(f"Niche {niche_id} inconnue ou sans profil.")
    row = AgentSession(niche_id=niche_id, messages=[])
    session.add(row)
    session.commit()
    return row


def _first_message(session: Session, niche: Niche, message: str) -> list[dict[str, Any]]:
    name, version = PROMPT
    memory = json.dumps(load_memory(session, niche), ensure_ascii=False, default=str)
    return [
        {"type": "text", "text": load_prompt(name, version).render_user({"memory": memory})},
        {"type": "text", "text": message},
    ]


def _stored_blocks(content: list[Any]) -> list[dict[str, Any]]:
    """Forme simple et sérialisable des blocs d'une réponse."""
    blocks: list[dict[str, Any]] = []
    for block in content:
        if block.type == "text" and block.text:
            blocks.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            blocks.append(
                {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
            )
    return blocks or [{"type": "text", "text": "(pas de réponse)"}]


def _text(content: list[Any]) -> str:
    return "\n".join(block.text for block in content if block.type == "text" and block.text).strip()


def _spent_since(session: Session, last_call_id: int) -> tuple[int, float]:
    tokens, cost = session.execute(
        select(
            func.coalesce(
                func.sum(LLMCall.input_tokens + LLMCall.output_tokens + LLMCall.cache_write_tokens),
                0,
            ),
            func.coalesce(func.sum(LLMCall.cost_usd), 0.0),
        ).where(LLMCall.id > last_call_id)
    ).one()
    return int(tokens), round(float(cost), 4)


def chat(
    session: Session,
    agent_session: AgentSession,
    message: str,
    claude: ClaudeClient,
    on_step: Callable[[Step], None] | None = None,
    embedder: Embedder | None = None,
    ingest_deps: IngestDeps | None = None,
) -> ChatResult:
    """Un tour de conversation : Claude enchaîne les outils jusqu'à sa réponse.

    Tout ce qui est fait reste fait : si le tour s'interrompt (budget, panne, trop d'étapes),
    l'historique est refermé proprement et `interrupted` en donne la raison.
    """
    settings = get_settings()
    niche = session.get(Niche, agent_session.niche_id)
    first_call = session.scalar(select(func.coalesce(func.max(LLMCall.id), 0)))
    context = ToolContext(
        session=session,
        niche=niche,
        claude=claude,
        session_id=agent_session.id,
        embedder=embedder,
        ingest_deps=ingest_deps,
    )
    result = ChatResult(session_id=agent_session.id, answer="")

    def show(step: Step) -> None:
        result.steps.append(step)
        if on_step:
            on_step(step)

    content = message if agent_session.messages else _first_message(session, niche, message)
    stored = [*agent_session.messages, {"role": "user", "content": content}]
    live = list(stored)  # Comme `stored`, mais avec les blocs bruts du tour en cours.
    name, version = PROMPT
    tools = api_tools()

    with logged_action(
        session,
        "chat",
        "chat_turn",
        {"message": message[:MAX_LOGGED_MESSAGE_CHARS]},
        agent_session.id,
    ) as action:
        try:
            for _ in range(settings.agent_max_tool_rounds):
                check_budget(session, TOKENS_CHAT_ROUND)
                turn = claude.call_tools(
                    prompt=name,
                    version=version,
                    model=settings.model_agent,
                    messages=live,
                    tools=tools,
                )
                stored.append({"role": "assistant", "content": _stored_blocks(turn.content)})
                live.append({"role": "assistant", "content": turn.content})
                text = _text(turn.content)

                calls = [block for block in turn.content if block.type == "tool_use"]
                if turn.stop_reason != "tool_use" or not calls:
                    result.answer = text
                    break

                if text:
                    show(Step("text", detail=text))
                results = []
                for call in calls:
                    show(Step("tool_start", call.name, json.dumps(call.input, ensure_ascii=False)))
                    outcome = run_tool(context, call.name, call.input)
                    show(Step("tool_end", call.name, outcome.content, outcome.is_error))
                    block: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": outcome.content,
                    }
                    if outcome.is_error:
                        block["is_error"] = True
                    results.append(block)
                reply = {"role": "user", "content": results}
                stored.append(reply)
                live.append(reply)
            else:
                result.interrupted = (
                    f"Arrêté après {settings.agent_max_tool_rounds} étapes (AGENT_MAX_TOOL_ROUNDS)."
                )
        except BudgetExceeded as exc:
            action.status = ActionStatus.REFUSED
            result.interrupted = str(exc)
        except ClaudeCredentialsError:
            raise
        except (ClaudeError, anthropic.APIError) as exc:
            session.rollback()
            logger.warning("Tour de conversation interrompu : %s", exc)
            action.status = ActionStatus.ERROR
            result.interrupted = f"Échec de l'appel à Claude : {exc}"

        if result.interrupted:
            result.answer = f"Je m'arrête ici. {result.interrupted}"
            # Claude attend une réponse après un message utilisateur : l'historique reste valide.
            if stored[-1]["role"] == "user":
                stored.append(
                    {"role": "assistant", "content": [{"type": "text", "text": result.answer}]}
                )

        agent_session.messages = stored
        session.commit()
        result.evidence, result.unverified = check_citations(session, result.answer)
        result.tokens, result.cost_usd = _spent_since(session, first_call)
        action.summary = result.answer

    return result
