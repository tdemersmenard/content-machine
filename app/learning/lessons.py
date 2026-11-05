"""Leçons : relues et réécrites tous les 5 nouveaux résultats."""

import json
from collections import Counter

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.base import utcnow
from app.db.models import (
    Lesson,
    LessonScope,
    Niche,
    Pattern,
    Performance,
    PostMortem,
    Script,
    SynthesisRun,
)
from app.generate.service import active_lessons
from app.learning.schemas import LessonDraft, LessonSynthesis
from app.llm.client import ClaudeClient

RESULTS_PER_SYNTHESIS = 5
MIN_RESULTS_TO_AVOID = 3  # Un seul résultat ne suffit pas à bannir un pattern.
PROMPT = ("lessons_synthesis", "v1")
REPLACED = "Remplacée par la nouvelle synthèse."


def _post_mortems(session: Session, niche: Niche) -> list[PostMortem]:
    return list(
        session.scalars(
            select(PostMortem)
            .join(Script, Script.id == PostMortem.script_id)
            .where(Script.niche_id == niche.id)
            .order_by(PostMortem.id)
        )
    )


def _result_count(session: Session, niche: Niche) -> int:
    return len({item.script_id for item in _post_mortems(session, niche)})


def _results_per_pattern(session: Session, niche: Niche) -> Counter[int]:
    rows = session.scalars(
        select(Script.pattern_ids).where(
            Script.niche_id == niche.id, Script.id.in_(select(PostMortem.script_id))
        )
    )
    return Counter(pattern_id for pattern_ids in rows for pattern_id in set(pattern_ids))


def _evidence_rows(session: Session, niche: Niche) -> list[dict]:
    rows = []
    for item in _post_mortems(session, niche):
        script = session.get(Script, item.script_id)
        performance = session.scalar(
            select(Performance).where(
                Performance.script_id == item.script_id,
                Performance.horizon_h == item.horizon_h,
            )
        )
        rows.append(
            {
                "video_id": item.video_id,
                "horizon_h": item.horizon_h,
                "composite": performance.composite if performance else None,
                "label": performance.label if performance else None,
                "indexes": performance.indexes if performance else None,
                "pattern_ids": script.pattern_ids,
                "experiment": script.content.get("experiment", False),
                "hook_type": script.content["hooks"][0]["hook_type"],
                "angle": script.content["angle"],
                "post_mortem": item.content,
            }
        )
    return rows


def _pattern_table(session: Session, counts: Counter[int]) -> list[dict]:
    patterns = session.scalars(select(Pattern).where(Pattern.id.in_(counts)))
    return [
        {
            "id": item.id,
            "type": item.type,
            "name": item.name,
            "score_alpha": round(item.score_alpha, 3),
            "score_beta": round(item.score_beta, 3),
            "results": counts[item.id],
        }
        for item in patterns
    ]


def _build_lesson(
    draft: LessonDraft, niche: Niche, video_ids: set[int], counts: Counter[int]
) -> Lesson | None:
    """Une leçon sans vidéo réelle à l'appui est écartée, quoi qu'en dise Claude."""
    evidence_ids = [item for item in draft.evidence_video_ids if item in video_ids]
    if not evidence_ids:
        return None
    avoid = [item for item in draft.avoid_pattern_ids if counts[item] >= MIN_RESULTS_TO_AVOID]
    scope = LessonScope(draft.scope)
    return Lesson(
        scope=scope,
        niche_id=niche.id if scope is LessonScope.NICHE else None,
        statement=draft.statement,
        evidence=[
            *[{"video_id": item} for item in evidence_ids],
            *[{"pattern_id": item, "effect": "avoid"} for item in avoid],
        ],
        confidence=draft.confidence,
    )


def synthesize_lessons(session: Session, niche: Niche, claude: ClaudeClient) -> list[Lesson]:
    """Réécrit la liste des leçons. Les anciennes sont désactivées, jamais supprimées."""
    current = active_lessons(session, niche.id)
    evidence = _evidence_rows(session, niche)
    counts = _results_per_pattern(session, niche)
    name, version = PROMPT
    result = claude.call_json(
        prompt=name,
        version=version,
        schema=LessonSynthesis,
        model=get_settings().model_analysis,
        variables={
            "results": json.dumps(evidence, ensure_ascii=False),
            "patterns": json.dumps(_pattern_table(session, counts), ensure_ascii=False),
            "lessons": json.dumps(
                [
                    {"id": item.id, "statement": item.statement, "confidence": item.confidence}
                    for item in current
                ],
                ensure_ascii=False,
            ),
        },
        effort="high",
    )

    video_ids = {row["video_id"] for row in evidence}
    built = [_build_lesson(item, niche, video_ids, counts) for item in result.data.lessons]
    new = [item for item in built if item is not None]
    reasons = {item.lesson_id: item.reason for item in result.data.contradicted}

    now = utcnow()
    for lesson in current:
        lesson.active = False
        lesson.deactivated_at = now
        lesson.deactivated_reason = reasons.get(lesson.id, REPLACED)
    session.add_all(new)
    session.add(
        SynthesisRun(
            niche_id=niche.id,
            results_count=_result_count(session, niche),
            prompt_version=result.prompt_version,
        )
    )
    session.commit()
    return new


def maybe_synthesize(
    session: Session, niche: Niche, claude: ClaudeClient, force: bool = False
) -> bool:
    """Lance la synthèse tous les 5 nouveaux résultats."""
    last = session.scalar(
        select(func.max(SynthesisRun.results_count)).where(SynthesisRun.niche_id == niche.id)
    )
    count = _result_count(session, niche)
    if count == 0 or (not force and count - (last or 0) < RESULTS_PER_SYNTHESIS):
        return False
    synthesize_lessons(session, niche, claude)
    return True
