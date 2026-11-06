"""Journal de tout ce que fait l'agent, et notifications."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.db.models import ActionStatus, AgentAction, Notification

logger = logging.getLogger(__name__)

MAX_SUMMARY_CHARS = 1000


@contextmanager
def logged_action(
    session: Session,
    source: str,
    action: str,
    input: dict[str, Any] | None = None,
    session_id: int | None = None,
) -> Iterator[AgentAction]:
    """Écrit l'action avant de la lancer : le journal montre aussi ce qui est en cours.

    Une exception est notée puis propagée. Le bloc peut remplir `summary` ou passer `status` à
    REFUSED.
    """
    row = AgentAction(source=source, action=action, input=input or {}, session_id=session_id)
    session.add(row)
    session.commit()
    row_id = row.id
    try:
        yield row
    except Exception as exc:
        session.rollback()
        row = session.get(AgentAction, row_id)
        row.status = ActionStatus.ERROR
        row.summary = str(exc)[:MAX_SUMMARY_CHARS]
        row.finished_at = utcnow()
        session.commit()
        raise
    if row.status is ActionStatus.RUNNING:
        row.status = ActionStatus.OK
    row.summary = (row.summary or "")[:MAX_SUMMARY_CHARS] or None
    row.finished_at = utcnow()
    session.commit()


def notify(
    session: Session,
    kind: str,
    title: str,
    body: str = "",
    niche_id: int | None = None,
    script_id: int | None = None,
    horizon_h: int | None = None,
) -> Notification:
    row = Notification(
        kind=kind,
        title=title,
        body=body,
        niche_id=niche_id,
        script_id=script_id,
        horizon_h=horizon_h,
    )
    session.add(row)
    session.commit()
    logger.info("Notification %s : %s", kind, title)
    return row
