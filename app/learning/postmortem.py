"""Diagnostic d'une vidéo publiée : ce qui était prévu, ce qui a été tourné, ce qui s'est passé."""

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Analysis, Performance, PostMortem, Script, Transcript
from app.learning.schemas import PostMortemResult
from app.llm.client import ClaudeClient

MIN_HORIZON_H = 72  # À 24 h, il est trop tôt pour conclure : on attend le relevé suivant.
PROMPT = ("post_mortem", "v1")


def run_post_mortem(
    session: Session, script: Script, row: Performance, claude: ClaudeClient
) -> PostMortem | None:
    """Un diagnostic par script et par horizon, à partir de 72 h, une fois le relevé noté."""
    if row.horizon_h < MIN_HORIZON_H or row.composite is None:
        return None
    existing = session.scalar(
        select(PostMortem).where(
            PostMortem.script_id == script.id, PostMortem.horizon_h == row.horizon_h
        )
    )
    if existing is not None:
        return existing

    transcript = session.get(Transcript, row.video_id)
    analysis = session.scalar(
        select(Analysis)
        .where(Analysis.video_id == row.video_id)
        .order_by(Analysis.id.desc())
        .limit(1)
    )
    name, version = PROMPT
    result = claude.call_json(
        prompt=name,
        version=version,
        schema=PostMortemResult,
        model=get_settings().model_analysis,
        variables={
            "script": json.dumps(script.content, ensure_ascii=False),
            "actual_transcript": transcript.full_text if transcript else "",
            "published_analysis": (
                json.dumps(analysis.json_result["analysis"], ensure_ascii=False) if analysis else ""
            ),
            "stats": json.dumps(
                {
                    "horizon_h": row.horizon_h,
                    "composite": row.composite,
                    "label": row.label,
                    "indexes": row.indexes,
                    "baseline_size": row.baseline_size,
                    "predicted_score": script.predicted_score,
                    "views": row.views,
                    "retention_3s": row.retention_3s,
                    "retention_50": row.retention_50,
                    "completion_rate": row.completion_rate,
                    "avg_watch_time_s": row.avg_watch_time_s,
                    "notes": row.notes,
                },
                ensure_ascii=False,
            ),
        },
        effort="high",
    )
    post_mortem = PostMortem(
        script_id=script.id,
        video_id=row.video_id,
        horizon_h=row.horizon_h,
        prompt_version=result.prompt_version,
        content=result.data.model_dump(mode="json"),
    )
    session.add(post_mortem)
    session.commit()
    return post_mortem
