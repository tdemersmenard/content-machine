"""Budget de tokens par jour, en heure locale, calculé depuis `llm_calls`.

Sont comptés les tokens d'entrée, de sortie et d'écriture de cache. Une lecture de cache coûte
20 fois moins cher qu'un token d'entrée, elle ne compte pas.

Le budget se vérifie avant de lancer une action, avec une estimation de ce qu'elle va consommer.
Une action lancée n'est pas interrompue : le dépassement est limité à une seule action. Le budget
s'applique à l'agent et aux routines, pas aux commandes lancées à la main comme `cm adapt`.
"""

import threading
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.base import utcnow
from app.db.models import LLMCall

# Moyennes mesurées sur les vrais appels de `llm_calls`, arrondies vers le haut.
TOKENS_CHAT_ROUND = 15_000
TOKENS_INGEST = 50_000
TOKENS_ANALYSIS = 130_000
TOKENS_SCRIPT = 100_000
TOKENS_ADAPT_BASE = 40_000
TOKENS_IDEAS = 30_000
TOKENS_POST_MORTEM = 15_000
TOKENS_SYNTHESIS = 15_000
TOKENS_WEEKLY_REPORT = 15_000

# Deux tâches ne doivent pas payer la même analyse. Valable dans le process de l'API.
paid_work = threading.Lock()


class BudgetExceeded(Exception):
    """Le budget du jour ne couvre pas l'action demandée."""


def local_zone() -> ZoneInfo:
    return ZoneInfo(get_settings().agent_timezone)


def local_now(now: datetime | None = None) -> datetime:
    return (now or utcnow()).astimezone(local_zone())


def day_start(now: datetime | None = None) -> datetime:
    midnight = datetime.combine(local_now(now).date(), time.min, tzinfo=local_zone())
    return midnight.astimezone(UTC)


def tokens_used(session: Session, now: datetime | None = None) -> int:
    billed = LLMCall.input_tokens + LLMCall.output_tokens + LLMCall.cache_write_tokens
    total = session.scalar(
        select(func.coalesce(func.sum(billed), 0)).where(LLMCall.created_at >= day_start(now))
    )
    return int(total)


def budget_status(session: Session, now: datetime | None = None) -> dict[str, int | float]:
    limit = get_settings().agent_daily_token_budget
    used = tokens_used(session, now)
    cost = session.scalar(
        select(func.coalesce(func.sum(LLMCall.cost_usd), 0.0)).where(
            LLMCall.created_at >= day_start(now)
        )
    )
    return {
        "limit": limit,
        "used": used,
        "remaining": max(limit - used, 0),
        "cost_usd": round(float(cost), 2),
    }


def can_spend(session: Session, tokens: int) -> bool:
    return tokens_used(session) + tokens <= get_settings().agent_daily_token_budget


def check_budget(session: Session, tokens: int = 0) -> None:
    if not can_spend(session, tokens):
        limit = get_settings().agent_daily_token_budget
        used = tokens_used(session)
        raise BudgetExceeded(
            f"Budget du jour insuffisant : {used} tokens utilisés sur {limit}, "
            f"il en faut environ {tokens} de plus. AGENT_DAILY_TOKEN_BUDGET permet de l'augmenter."
        )
