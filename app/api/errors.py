"""Traduction des erreurs de Claude en réponses HTTP."""

from collections.abc import Iterator
from contextlib import contextmanager

import anthropic
from fastapi import HTTPException

from app.llm.errors import ClaudeCredentialsError, ClaudeError


@contextmanager
def claude_errors() -> Iterator[None]:
    try:
        yield
    except ClaudeCredentialsError as exc:
        raise HTTPException(503, str(exc)) from exc
    except (ClaudeError, anthropic.APIError) as exc:
        raise HTTPException(502, f"Échec de l'appel à Claude : {exc}") from exc
