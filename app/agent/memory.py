"""Ce que l'agent sait au début d'une session."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Niche, Performance, Script, WeeklyReport
from app.generate.service import active_lessons
from app.learning.report import build_report

RECENT_SCRIPTS = 10


def _result(session: Session, script: Script) -> dict[str, Any] | None:
    """Le relevé le plus tardif du script, noté ou non."""
    row = session.scalar(
        select(Performance)
        .where(Performance.script_id == script.id)
        .order_by(Performance.horizon_h.desc())
        .limit(1)
    )
    if row is None:
        return None
    return {
        "horizon_h": row.horizon_h,
        "views": row.views,
        "composite": row.composite,
        "label": row.label,
    }


def recent_scripts(session: Session, niche_id: int) -> list[dict[str, Any]]:
    scripts = session.scalars(
        select(Script)
        .where(Script.niche_id == niche_id)
        .order_by(Script.id.desc())
        .limit(RECENT_SCRIPTS)
    )
    return [
        {
            "id": item.id,
            "title": item.content["internal_title"],
            "angle": item.content["angle"],
            "status": item.status.value,
            "experiment": item.content.get("experiment", False),
            "published_video_id": item.published_video_id,
            "result": _result(session, item),
        }
        for item in scripts
    ]


def load_memory(session: Session, niche: Niche) -> dict[str, Any]:
    """Profil, leçons actives, dernier rapport et 10 derniers scripts."""
    weekly = session.scalar(
        select(WeeklyReport)
        .where(WeeklyReport.niche_id == niche.id)
        .order_by(WeeklyReport.week_start.desc())
        .limit(1)
    )
    return {
        "niche": {"id": niche.id, "name": niche.name, "profile": niche.profile},
        "active_lessons": [
            {
                "id": item.id,
                "statement": item.statement,
                "confidence": item.confidence,
                "evidence": item.evidence,
            }
            for item in active_lessons(session, niche.id)
        ],
        "latest_weekly_report": weekly.markdown if weekly else None,
        "learning_report": build_report(session, niche.id),
        "recent_scripts": recent_scripts(session, niche.id),
    }
