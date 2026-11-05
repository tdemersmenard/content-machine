"""Wrapper Claude : point d'entrée unique pour tous les appels à l'API.

- prompts chargés depuis app/prompts/ (jamais en dur) ;
- sortie JSON contrainte par schéma, puis validée par Pydantic, avec un seul retry ;
- retries réseau, 429 et 5xx délégués au SDK ;
- chaque tentative est journalisée (tokens, coût, prompt_version).
"""

import base64
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import anthropic
from anthropic import Anthropic, transform_schema
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.db.models import LLMCall, LLMCallStatus
from app.db.session import get_session_factory
from app.llm.errors import (
    ClaudeCredentialsError,
    ClaudeInvalidOutputError,
    ClaudeRefusalError,
    ClaudeTruncatedError,
)
from app.llm.pricing import TokenUsage, compute_cost
from app.llm.prompts import Prompt, load_prompt

logger = logging.getLogger(__name__)

SDK_MAX_RETRIES = 4
MAX_ATTEMPTS = 2  # L'appel initial, plus un retry si la sortie est invalide.
DEFAULT_MAX_TOKENS = 64_000

# Modèles qui acceptent le fallback serveur en cas de refus (pas Haiku, par exemple).
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)

IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True)
class LLMCallRecord:
    """Une tentative d'appel. Les champs correspondent à la table llm_calls."""

    prompt_version: str
    prompt_hash: str
    model_requested: str
    attempt: int
    status: LLMCallStatus
    latency_ms: int
    model_served: str | None = None
    request_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None
    raw_output: str | None = None


CallLogger = Callable[[LLMCallRecord], None]


@dataclass(frozen=True)
class ClaudeResult[T: BaseModel]:
    data: T
    model: str
    prompt_version: str
    attempts: int
    cost_usd: float | None


def db_call_logger(record: LLMCallRecord) -> None:
    """Écrit dans sa propre session : la trace survit à un rollback de l'appelant."""
    with get_session_factory()() as session:
        session.add(LLMCall(**vars(record)))
        session.commit()


def extract_usages(message: Any) -> list[TokenUsage]:
    """Tokens par modèle. En cas de fallback, `usage.iterations` détaille chaque modèle."""
    usage = message.usage
    entries = [
        (entry, entry.model or message.model)
        for entry in (usage.iterations or [])
        if entry.type in ("message", "fallback_message")
    ] or [(usage, message.model)]

    return [
        TokenUsage(
            model=model,
            input_tokens=entry.input_tokens or 0,
            output_tokens=entry.output_tokens or 0,
            cache_read_tokens=entry.cache_read_input_tokens or 0,
            cache_write_tokens=entry.cache_creation_input_tokens or 0,
        )
        for entry, model in entries
    ]


def _image_block(path: Path) -> dict[str, Any]:
    media_type = IMAGE_MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise ValueError(f"Format d'image non supporté : {path}")
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def _correction_message(error: ValidationError) -> str:
    return (
        "Ta réponse précédente ne respecte pas le schéma attendu. Erreurs de validation :\n"
        f"{error}\n\nRenvoie uniquement l'objet JSON complet et corrigé."
    )


class ClaudeClient:
    def __init__(self, sdk: Anthropic, call_logger: CallLogger = db_call_logger) -> None:
        self._sdk = sdk
        self._call_logger = call_logger

    def call_json[T: BaseModel](
        self,
        *,
        prompt: str,
        version: str,
        schema: type[T],
        model: str,
        variables: dict[str, Any] | None = None,
        images: list[Path] | None = None,
        effort: Effort = "high",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> ClaudeResult[T]:
        """Appelle Claude et renvoie une sortie validée par `schema`.

        Lève ClaudeInvalidOutputError, ClaudeRefusalError, ClaudeTruncatedError, ou
        l'erreur du SDK (anthropic.APIError) une fois ses retries épuisés.
        """
        loaded = load_prompt(prompt, version)
        content: list[dict[str, Any]] = [_image_block(path) for path in images or []]
        content.append({"type": "text", "text": loaded.render_user(variables)})
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]

        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": loaded.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "output_config": {
                "effort": effort,
                "format": {"type": "json_schema", "schema": transform_schema(schema)},
            },
        }
        if model in FALLBACK_MODELS:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"

        total_cost: float | None = 0.0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            message, request_id, latency_ms = self._send(loaded, params, messages, model, attempt)
            usages = extract_usages(message)
            cost = compute_cost(usages)
            total_cost = None if total_cost is None or cost is None else total_cost + cost

            fields: dict[str, Any] = {
                "prompt_version": loaded.prompt_version,
                "prompt_hash": loaded.content_hash,
                "model_requested": model,
                "model_served": message.model,
                "request_id": request_id,
                "attempt": attempt,
                "latency_ms": latency_ms,
                "input_tokens": sum(u.input_tokens for u in usages),
                "output_tokens": sum(u.output_tokens for u in usages),
                "cache_read_tokens": sum(u.cache_read_tokens for u in usages),
                "cache_write_tokens": sum(u.cache_write_tokens for u in usages),
                "cost_usd": cost,
            }

            if message.stop_reason == "refusal":
                details = message.stop_details
                category = details.category if details else None
                explanation = details.explanation if details else None
                error = f"Refus ({category}) : {explanation}"
                self._record(LLMCallRecord(**fields, status=LLMCallStatus.REFUSAL, error=error))
                raise ClaudeRefusalError(error, category=category)

            text = "".join(block.text for block in message.content if block.type == "text")

            if message.stop_reason == "max_tokens":
                error = f"Sortie tronquée à max_tokens={max_tokens}"
                self._record(
                    LLMCallRecord(
                        **fields, status=LLMCallStatus.TRUNCATED, error=error, raw_output=text
                    )
                )
                raise ClaudeTruncatedError(error)

            try:
                data = schema.model_validate_json(text)
            except ValidationError as exc:
                self._record(
                    LLMCallRecord(
                        **fields,
                        status=LLMCallStatus.INVALID_OUTPUT,
                        error=str(exc),
                        raw_output=text,
                    )
                )
                if attempt == MAX_ATTEMPTS:
                    raise ClaudeInvalidOutputError(
                        f"{loaded.prompt_version} : sortie invalide après {attempt} tentatives",
                        raw_output=text,
                    ) from exc
                # Historique en ajout seul, blocs renvoyés tels quels (thinking compris).
                messages = [
                    *messages,
                    {"role": "assistant", "content": message.content},
                    {"role": "user", "content": _correction_message(exc)},
                ]
                continue

            self._record(LLMCallRecord(**fields, status=LLMCallStatus.OK))
            return ClaudeResult(
                data=data,
                model=message.model,
                prompt_version=loaded.prompt_version,
                attempts=attempt,
                cost_usd=total_cost,
            )

        raise AssertionError("Boucle de tentatives terminée sans résultat")  # pragma: no cover

    def _send(
        self,
        prompt: Prompt,
        params: dict[str, Any],
        messages: list[dict[str, Any]],
        model: str,
        attempt: int,
    ) -> tuple[Any, str | None, int]:
        started = time.perf_counter()

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started) * 1000)

        try:
            with self._sdk.beta.messages.stream(**params, messages=messages) as stream:
                message = stream.get_final_message()
                request_id = getattr(stream, "request_id", None)
        except TypeError as exc:
            # Le SDK signale l'absence d'identifiants par un TypeError avant tout envoi.
            if "authentication" not in str(exc):
                raise
            raise ClaudeCredentialsError(
                "Aucune clé API : renseigne ANTHROPIC_API_KEY dans le fichier .env."
            ) from exc
        except anthropic.APIError as exc:
            self._record(
                LLMCallRecord(
                    prompt_version=prompt.prompt_version,
                    prompt_hash=prompt.content_hash,
                    model_requested=model,
                    attempt=attempt,
                    status=LLMCallStatus.ERROR,
                    latency_ms=elapsed_ms(),
                    request_id=getattr(exc, "request_id", None),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        return message, request_id, elapsed_ms()

    def _record(self, record: LLMCallRecord) -> None:
        logger.info(
            "claude_call prompt=%s attempt=%d status=%s model=%s in=%d out=%d "
            "cache_read=%d cache_write=%d cost_usd=%s latency_ms=%d",
            record.prompt_version,
            record.attempt,
            record.status.value,
            record.model_served or record.model_requested,
            record.input_tokens,
            record.output_tokens,
            record.cache_read_tokens,
            record.cache_write_tokens,
            record.cost_usd,
            record.latency_ms,
        )
        try:
            self._call_logger(record)
        except Exception:
            # Un échec de journalisation ne doit pas faire perdre un appel déjà payé.
            logger.exception("Impossible de journaliser l'appel %s", record.prompt_version)


@lru_cache
def get_claude() -> ClaudeClient:
    key = get_settings().anthropic_api_key
    # Sans clé, le SDK résout lui-même les identifiants (profil `ant auth login`, etc.).
    kwargs = {"api_key": key.get_secret_value()} if key else {}
    return ClaudeClient(Anthropic(max_retries=SDK_MAX_RETRIES, **kwargs))
