"""Rapport de suivi : tout est calculé, aucun appel à Claude."""

from collections import defaultdict
from statistics import mean, median
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Lesson, Pattern, PatternScore, Script
from app.generate.service import active_lessons
from app.learning.scoring import SUCCESS_ABOVE
from app.learning.service import scored_results

MIN_FOR_CALIBRATION = 3
TOP_PATTERNS = 5


def prediction_accuracy(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """Compare predicted_score au composite réel.

    `direction_accuracy` : part des vidéos où « prédite au-dessus de la médiane » et
    « a battu la baseline » concordent. 0,5 équivaut au hasard.
    """
    result: dict[str, Any] = {"sample": len(pairs), "correlation": None, "direction_accuracy": None}
    if len(pairs) < MIN_FOR_CALIBRATION:
        return result

    predicted = [item[0] for item in pairs]
    actual = [item[1] for item in pairs]
    if len(set(predicted)) > 1 and len(set(actual)) > 1:
        result["correlation"] = round(float(np.corrcoef(predicted, actual)[0, 1]), 3)
    middle = median(predicted)
    hits = sum(
        (guess >= middle) == (real > SUCCESS_ABOVE)
        for guess, real in zip(predicted, actual, strict=True)
    )
    result["direction_accuracy"] = round(hits / len(pairs), 3)
    return result


def _timeline(results: list[tuple[Script, Any]]) -> list[dict[str, Any]]:
    """Composite moyen par mois."""
    months: dict[str, list[float]] = defaultdict(list)
    for _, row in results:
        months[row.measured_at.strftime("%Y-%m")].append(row.composite)
    return [
        {"month": month, "videos": len(values), "avg_composite": round(mean(values), 3)}
        for month, values in sorted(months.items())
    ]


def _pattern_ranking(session: Session, niche_id: int | None) -> dict[str, list[dict[str, Any]]]:
    patterns = {item.id: item for item in session.scalars(select(Pattern))}
    if niche_id is None:
        scores = [(item.id, item.score_alpha, item.score_beta) for item in patterns.values()]
    else:
        rows = session.scalars(select(PatternScore).where(PatternScore.niche_id == niche_id))
        scores = [(item.pattern_id, item.score_alpha, item.score_beta) for item in rows]

    tested = [
        {
            "id": pattern_id,
            "name": patterns[pattern_id].name,
            "type": patterns[pattern_id].type,
            "score": round(alpha / (alpha + beta), 3),
            "weighted_results": round(alpha + beta - 2, 2),
        }
        for pattern_id, alpha, beta in scores
        if alpha + beta > 2
    ]
    tested.sort(key=lambda item: item["score"], reverse=True)
    return {"top": tested[:TOP_PATTERNS], "flop": tested[::-1][:TOP_PATTERNS]}


def _lesson_out(lesson: Lesson) -> dict[str, Any]:
    return {
        "id": lesson.id,
        "statement": lesson.statement,
        "confidence": lesson.confidence,
        "scope": lesson.scope.value,
        "videos": [item["video_id"] for item in lesson.evidence if "video_id" in item],
    }


def build_report(session: Session, niche_id: int | None = None) -> dict[str, Any]:
    results = scored_results(session)
    if niche_id is not None:
        results = [item for item in results if item[0].niche_id == niche_id]

    pairs = [
        (script.predicted_score, row.composite)
        for script, row in results
        if script.predicted_score is not None
    ]
    labels: dict[str, int] = defaultdict(int)
    for _, row in results:
        labels[row.label] += 1

    lessons = (
        active_lessons(session, niche_id)
        if niche_id is not None
        else list(session.scalars(select(Lesson).where(Lesson.active.is_(True))))
    )
    return {
        "scored_videos": len(results),
        "labels": dict(labels),
        "composite_by_month": _timeline(results),
        "patterns": _pattern_ranking(session, niche_id),
        "prediction": prediction_accuracy(pairs),
        "active_lessons": [_lesson_out(item) for item in lessons],
    }
