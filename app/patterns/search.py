"""Recherche et classement dans la bibliothèque de patterns."""

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Niche, Pattern, PatternType, Performance, Script, Video, VideoPattern
from app.patterns.embeddings import Embedder, embed_texts, from_bytes
from app.patterns.schemas import PatternDetail, PatternOut, PatternSearchHit, PatternVideoOut


def score_mean(pattern: Pattern) -> float:
    """Moyenne de la loi Beta. Vaut 0,5 tant que la boucle d'apprentissage n'a rien mesuré."""
    return pattern.score_alpha / (pattern.score_alpha + pattern.score_beta)


def _fields(pattern: Pattern) -> dict:
    return {
        "id": pattern.id,
        "type": pattern.type,
        "name": pattern.name,
        "description": pattern.description,
        "template": pattern.template,
        "when_to_use": pattern.when_to_use,
        "when_not_to_use": pattern.when_not_to_use,
        "examples": pattern.examples,
        "score_alpha": pattern.score_alpha,
        "score_beta": pattern.score_beta,
        "score_mean": score_mean(pattern),
        "example_count": len(pattern.examples),
        "times_used": pattern.times_used,
        "created_at": pattern.created_at,
        "updated_at": pattern.updated_at,
    }


def to_out(pattern: Pattern) -> PatternOut:
    return PatternOut(**_fields(pattern))


def to_detail(session: Session, pattern: Pattern) -> PatternDetail:
    links = session.scalars(select(VideoPattern).where(VideoPattern.pattern_id == pattern.id))
    return PatternDetail(
        **_fields(pattern),
        videos=[PatternVideoOut.model_validate(link) for link in links],
    )


def search_patterns(
    session: Session,
    query: str,
    type: PatternType | None = None,
    k: int = 10,
    embedder: Embedder | None = None,
) -> list[PatternSearchHit]:
    """Recherche sémantique, du plus proche au moins proche."""
    statement = select(Pattern).where(
        Pattern.embedding.is_not(None),
        Pattern.embedding_model == get_settings().embedding_model,
    )
    if type is not None:
        statement = statement.where(Pattern.type == type.value)
    patterns = session.scalars(statement).all()
    if not patterns:
        return []

    vector = (embedder or embed_texts)([query])[0]
    scored = [(item, float(from_bytes(item.embedding) @ vector)) for item in patterns]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [PatternSearchHit(**_fields(item), similarity=score) for item, score in scored[:k]]


def _niche_pattern_ids(session: Session, niche_id: int) -> set[int]:
    """Patterns vus dans des vidéos de cette niche, ou utilisés dans ses scripts."""
    niche = session.get(Niche, niche_id)
    if niche is None:
        raise LookupError(f"Niche {niche_id} inconnue.")

    from_videos = session.scalars(
        select(VideoPattern.pattern_id)
        .join(Video, Video.id == VideoPattern.video_id)
        .where(Video.niche_source == niche.name)
    )
    from_scripts = session.scalars(select(Script.pattern_ids).where(Script.niche_id == niche_id))
    return {*from_videos, *[item for ids in from_scripts for item in ids]}


def top_patterns(
    session: Session,
    type: PatternType | None = None,
    niche_id: int | None = None,
    limit: int = 50,
) -> list[Pattern]:
    """Classement par moyenne de la Beta, puis par nombre d'exemples.

    Le score reste global : `niche_id` filtre les patterns, il n'y a pas encore de score par niche.
    """
    statement = select(Pattern)
    if type is not None:
        statement = statement.where(Pattern.type == type.value)
    patterns = list(session.scalars(statement))

    if niche_id is not None:
        allowed = _niche_pattern_ids(session, niche_id)
        patterns = [item for item in patterns if item.id in allowed]

    patterns.sort(key=lambda item: (-score_mean(item), -len(item.examples), item.id))
    return patterns[:limit]


@dataclass(frozen=True)
class Combination:
    pattern_ids: tuple[int, int]
    pattern_names: tuple[str, str]
    videos: int
    weighted_score: float  # Somme des percentiles de performance des vidéos concernées.
    avg_percentile: float
    lift: float  # Plus de 1 : les deux patterns se croisent plus que le hasard.


def _video_views(session: Session) -> dict[int, int]:
    """Vues connues par vidéo. Une vidéo sans vues mesurées est exclue, jamais estimée."""
    views = {
        video_id: value
        for video_id, value in session.execute(
            select(Video.id, Video.views).where(
                Video.views.is_not(None), Video.duplicate_of_id.is_(None)
            )
        )
    }
    measured = session.execute(
        select(Performance.video_id, Performance.views)
        .where(Performance.views.is_not(None))
        .order_by(Performance.measured_at)
    )
    views.update(dict(measured.all()))  # Le dernier relevé fait foi.
    return views


def winning_combinations(session: Session, k: int = 10, min_videos: int = 2) -> list[Combination]:
    """Paires de patterns fréquentes dans les vidéos performantes.

    Chaque vidéo pèse son percentile de vues dans la bibliothèque : le rang résiste mieux qu'un
    nombre de vues brut à une vidéo qui en a 1000 fois plus que les autres.
    """
    views = _video_views(session)
    if not views:
        return []
    ordered = sorted(views.values())
    percentile = {
        video_id: bisect_right(ordered, value) / len(ordered) for video_id, value in views.items()
    }

    by_video: dict[int, set[int]] = defaultdict(set)
    links = session.execute(select(VideoPattern.video_id, VideoPattern.pattern_id))
    for video_id, pattern_id in links:
        if video_id in percentile:
            by_video[video_id].add(pattern_id)

    singles: Counter[int] = Counter()
    pairs: Counter[tuple[int, int]] = Counter()
    weights: Counter[tuple[int, int]] = Counter()
    for video_id, pattern_ids in by_video.items():
        singles.update(pattern_ids)
        for pair in combinations(sorted(pattern_ids), 2):
            pairs[pair] += 1
            weights[pair] += percentile[video_id]

    total = len(by_video)
    kept = [pair for pair, count in pairs.items() if count >= min_videos]
    kept.sort(key=lambda pair: weights[pair], reverse=True)

    names = dict(session.execute(select(Pattern.id, Pattern.name)).all())
    return [
        Combination(
            pattern_ids=pair,
            pattern_names=(names[pair[0]], names[pair[1]]),
            videos=pairs[pair],
            weighted_score=round(weights[pair], 4),
            avg_percentile=round(weights[pair] / pairs[pair], 4),
            lift=round(pairs[pair] * total / (singles[pair[0]] * singles[pair[1]]), 4),
        )
        for pair in kept[:k]
    ]
