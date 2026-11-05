"""Enregistrement des stats, scores normalisés et mise à jour des patterns."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.base import utcnow
from app.db.models import (
    Niche,
    Pattern,
    PatternScore,
    Performance,
    PostMortem,
    Script,
    ScriptStatus,
    Video,
)
from app.generate.service import update_script
from app.learning.lessons import maybe_synthesize
from app.learning.postmortem import run_post_mortem
from app.learning.schemas import PerformanceInput
from app.learning.scoring import (
    METRICS,
    SUCCESS_ABOVE,
    baselines,
    classify,
    composite_score,
    decay_weight,
    performance_indexes,
    weights_for,
)
from app.llm.client import ClaudeClient
from app.niche.profile import load_profile

logger = logging.getLogger(__name__)


@dataclass
class RecordResult:
    performance: Performance
    post_mortem: PostMortem | None = None
    lessons_updated: bool = False
    warnings: list[str] = field(default_factory=list)


def _row_values(row: Performance) -> dict[str, float | None]:
    return {metric: getattr(row, metric) for metric in METRICS}


def _weights(session: Session, row: Performance) -> dict[str, float]:
    script = session.get(Script, row.script_id) if row.script_id else None
    niche = session.get(Niche, script.niche_id) if script else None
    if niche is None or niche.profile is None:
        return weights_for(None)
    profile = load_profile(niche)
    return weights_for(profile.identity.goal, profile.score_weights)


def rescore_horizon(session: Session, horizon_h: int) -> None:
    """Recalcule tous les scores d'un horizon : un relevé ajouté en retard reste juste.

    Chaque relevé est comparé aux relevés qui le précèdent, jamais à lui-même.
    """
    rows = list(
        session.scalars(
            select(Performance)
            .where(Performance.horizon_h == horizon_h)
            .order_by(Performance.measured_at, Performance.id)
        )
    )
    minimum = get_settings().learning_min_baseline
    history: list[dict[str, float | None]] = []
    for row in rows:
        values = _row_values(row)
        row.indexes = row.composite = row.label = row.baseline_size = None
        if len(history) >= minimum:
            baseline = baselines(history, minimum)
            indexes = performance_indexes(values, baseline)
            composite = composite_score(indexes, _weights(session, row))
            if composite is not None:
                row.indexes = indexes
                row.composite = composite
                row.label = classify(composite)
                row.baseline_size = min(len(history), 20)
        history.append(values)


def scored_results(session: Session) -> list[tuple[Script, Performance]]:
    """Un résultat par script : le relevé noté de l'horizon le plus long.

    Une vidéo ne vote qu'une fois, même avec trois relevés.
    """
    rows = session.execute(
        select(Script, Performance)
        .join(Performance, Performance.script_id == Script.id)
        .where(Performance.composite.is_not(None))
        .order_by(Performance.horizon_h)
    ).all()
    latest = {script.id: (script, row) for script, row in rows}
    return list(latest.values())


def rebuild_pattern_scores(session: Session, now: datetime | None = None) -> None:
    """Reconstruit alpha et beta depuis tous les résultats, avec le decay temporel.

    Équivalent à alpha += succès et beta += échec, mais une correction de stats ne laisse
    aucune trace fausse, et un résultat ancien pèse moins.
    """
    now = now or utcnow()
    half_life = get_settings().learning_half_life_days
    known = set(session.scalars(select(Pattern.id)))
    overall: dict[int, list[float]] = defaultdict(lambda: [1.0, 1.0])
    by_niche: dict[tuple[int, int], list[float]] = defaultdict(lambda: [1.0, 1.0])

    for script, row in scored_results(session):
        age_days = (now - row.measured_at).total_seconds() / 86400
        weight = decay_weight(age_days, half_life)
        side = 0 if row.composite > SUCCESS_ABOVE else 1
        for pattern_id in set(script.pattern_ids) & known:
            overall[pattern_id][side] += weight
            by_niche[(pattern_id, script.niche_id)][side] += weight

    for pattern in session.scalars(select(Pattern)):
        pattern.score_alpha, pattern.score_beta = overall.get(pattern.id, (1.0, 1.0))

    session.execute(delete(PatternScore))
    session.add_all(
        PatternScore(pattern_id=pattern_id, niche_id=niche_id, score_alpha=a, score_beta=b)
        for (pattern_id, niche_id), (a, b) in by_niche.items()
    )


def _ensure_video(session: Session, script: Script, url: str | None) -> int:
    """La vidéo publiée du script. Si elle n'a pas encore été ingérée, on crée sa fiche."""
    if script.published_video_id is not None:
        return script.published_video_id

    video = session.scalar(select(Video).where(Video.source_url == url)) if url else None
    if video is None:
        niche = session.get(Niche, script.niche_id)
        video = Video(source_url=url, is_own_content=True, niche_source=niche.name)
        session.add(video)
        session.flush()
    script.published_video_id = video.id
    return video.id


def save_performance(
    session: Session, script: Script, data: PerformanceInput, rebuild: bool = True
) -> Performance:
    """Enregistre un relevé et recalcule les scores. Refaire un horizon corrige le relevé."""
    video_id = _ensure_video(session, script, data.video_url)
    row = session.scalar(
        select(Performance).where(
            Performance.script_id == script.id, Performance.horizon_h == data.horizon_h
        )
    )
    if row is None:
        row = Performance(script_id=script.id, video_id=video_id, horizon_h=data.horizon_h)
        session.add(row)

    row.measured_at = data.measured_at or utcnow()
    row.hours_since_post = float(data.horizon_h)
    row.notes = data.notes
    for metric in METRICS:
        setattr(row, metric, getattr(data, metric))
    session.flush()

    rescore_horizon(session, data.horizon_h)
    if rebuild:
        rebuild_pattern_scores(session)
    session.commit()

    if script.status is not ScriptStatus.PUBLISHED:
        update_script(session, script, status=ScriptStatus.PUBLISHED)
    return row


def run_learning(
    session: Session, script: Script, row: Performance, claude: ClaudeClient
) -> tuple[PostMortem | None, bool, list[str]]:
    """Post-mortem et leçons. Un échec ici ne doit pas faire perdre les stats enregistrées."""
    warnings: list[str] = []
    post_mortem = None
    updated = False
    try:
        post_mortem = run_post_mortem(session, script, row, claude)
    except Exception as exc:
        logger.exception("Post-mortem échoué pour le script %d", script.id)
        session.rollback()
        warnings.append(f"Post-mortem non généré : {exc}")
    try:
        updated = maybe_synthesize(session, session.get(Niche, script.niche_id), claude)
    except Exception as exc:
        logger.exception("Synthèse des leçons échouée")
        session.rollback()
        warnings.append(f"Leçons non mises à jour : {exc}")
    return post_mortem, updated, warnings


def record_performance(
    session: Session, script: Script, data: PerformanceInput, claude: ClaudeClient
) -> RecordResult:
    row = save_performance(session, script, data)
    post_mortem, updated, warnings = run_learning(session, script, row, claude)
    return RecordResult(row, post_mortem, updated, warnings)
