"""Extraction des patterns d'une analyse et fusion dans la bibliothèque.

Les candidats sont gardés sur la ligne `analyses` et traités un par un : une reprise après
échec ne repaie ni l'extraction, ni les fusions déjà faites.
"""

import json
import logging

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analysis.context import build_metadata, compute_metrics
from app.analysis.evidence import EvidenceIndex, build_index, check_evidence
from app.analysis.report import clock
from app.analysis.schemas import Evidence
from app.config import get_settings
from app.db.base import utcnow
from app.db.models import Analysis, Pattern, Video, VideoPattern
from app.llm.client import ClaudeClient, get_claude
from app.patterns.embeddings import Embedder, embed_texts, from_bytes, to_bytes
from app.patterns.schemas import (
    ExtractedPatterns,
    MergeDecision,
    PatternCandidate,
    embedding_text,
)

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = ("pattern_extract", "v1")
MERGE_PROMPT = ("pattern_merge", "v1")
MAX_MERGE_CHECKS = 3  # Nombre maximal de voisins soumis à Claude pour un candidat.


def find_similar(
    session: Session, pattern_type: str, vector: np.ndarray, threshold: float
) -> list[tuple[Pattern, float]]:
    """Patterns du même type au-dessus du seuil, du plus proche au moins proche."""
    model = get_settings().embedding_model
    patterns = session.scalars(
        select(Pattern).where(
            Pattern.type == pattern_type,
            Pattern.embedding.is_not(None),
            Pattern.embedding_model == model,
        )
    ).all()
    scored = [(item, float(from_bytes(item.embedding) @ vector)) for item in patterns]
    close = [(item, score) for item, score in scored if score > threshold]
    return sorted(close, key=lambda pair: pair[1], reverse=True)


def _same_mechanism(
    claude: ClaudeClient, candidate: PatternCandidate, existing: Pattern
) -> MergeDecision:
    name, version = MERGE_PROMPT
    return claude.call_json(
        prompt=name,
        version=version,
        schema=MergeDecision,
        model=get_settings().model_analysis,
        variables={
            "existing": json.dumps(
                {
                    "name": existing.name,
                    "description": existing.description,
                    "template": existing.template,
                },
                ensure_ascii=False,
            ),
            "candidate": json.dumps(
                {
                    "name": candidate.name,
                    "description": candidate.description,
                    "template": candidate.template,
                },
                ensure_ascii=False,
            ),
        },
        effort="medium",
    ).data


def _add_examples(pattern: Pattern, video_id: int, evidence: list[Evidence]) -> None:
    known = {(item["video_id"], item["timestamp"]) for item in pattern.examples}
    new = [
        {"video_id": video_id, "excerpt": item.excerpt, "timestamp": item.start_s}
        for item in evidence
        if (video_id, item.start_s) not in known
    ]
    pattern.examples = [*pattern.examples, *new]


def _integrate(
    session: Session,
    candidate: PatternCandidate,
    evidence: list[Evidence],
    video_id: int,
    claude: ClaudeClient,
    embedder: Embedder,
) -> Pattern:
    settings = get_settings()
    vector = embedder([candidate.embedding_text()])[0]

    neighbours = find_similar(
        session, candidate.type.value, vector, settings.pattern_merge_threshold
    )
    for existing, _ in neighbours[:MAX_MERGE_CHECKS]:
        decision = _same_mechanism(claude, candidate, existing)
        if decision.same_mechanism:
            existing.description = decision.merged_description
            existing.embedding = to_bytes(
                embedder([embedding_text(existing.name, existing.description, existing.template)])[
                    0
                ]
            )
            _add_examples(existing, video_id, evidence)
            return existing

    pattern = Pattern(
        type=candidate.type.value,
        name=candidate.name,
        description=candidate.description,
        template=candidate.template,
        when_to_use=candidate.when_to_use,
        when_not_to_use=candidate.when_not_to_use,
        embedding=to_bytes(vector),
        embedding_model=settings.embedding_model,
        prompt_version="/".join(EXTRACT_PROMPT),
        examples=[],
    )
    _add_examples(pattern, video_id, evidence)
    session.add(pattern)
    session.flush()
    return pattern


def _link(
    session: Session, video_id: int, pattern: Pattern, role: str, evidence: list[Evidence]
) -> None:
    proof = " | ".join(f"[{clock(item.start_s)}] {item.excerpt}" for item in evidence)
    session.merge(VideoPattern(video_id=video_id, pattern_id=pattern.id, role=role, evidence=proof))


def _valid_evidence(candidate: PatternCandidate, index: EvidenceIndex) -> list[Evidence]:
    return [item for item in candidate.evidence if check_evidence(item, index)[0]]


def _linked_patterns(session: Session, video_id: int) -> list[Pattern]:
    return list(
        session.scalars(
            select(Pattern)
            .join(VideoPattern, VideoPattern.pattern_id == Pattern.id)
            .where(VideoPattern.video_id == video_id)
            .distinct()
            .order_by(Pattern.id)
        )
    )


def extract_patterns(
    session: Session,
    analysis_id: int,
    claude: ClaudeClient | None = None,
    embedder: Embedder | None = None,
) -> list[Pattern]:
    """Extrait les patterns d'une analyse. Sans effet si c'est déjà fait."""
    analysis = session.get(Analysis, analysis_id)
    if analysis is None:
        raise LookupError(f"Analyse {analysis_id} inconnue.")
    if analysis.patterns_extracted_at is not None:
        return _linked_patterns(session, analysis.video_id)

    claude = claude or get_claude()
    embedder = embedder or embed_texts

    if analysis.pattern_candidates is None:
        name, version = EXTRACT_PROMPT
        result = claude.call_json(
            prompt=name,
            version=version,
            schema=ExtractedPatterns,
            model=get_settings().model_analysis,
            variables={
                "analysis": json.dumps(analysis.json_result["analysis"], ensure_ascii=False)
            },
            effort="high",
        )
        analysis.pattern_candidates = [
            candidate.model_dump(mode="json") for candidate in result.data.candidates
        ]
        session.commit()

    video = session.get(Video, analysis.video_id)
    root = get_settings().data_dir.resolve()
    timeline = json.loads((root / video.timeline_path).read_text("utf-8"))
    index = build_index(timeline, build_metadata(video, compute_metrics(video)))

    for position in range(analysis.pattern_candidates_done, len(analysis.pattern_candidates)):
        candidate = PatternCandidate.model_validate(analysis.pattern_candidates[position])
        evidence = _valid_evidence(candidate, index)
        if evidence:
            pattern = _integrate(session, candidate, evidence, video.id, claude, embedder)
            _link(session, video.id, pattern, candidate.role, evidence)
        else:
            logger.warning("Candidat sans preuve valable, ignoré : %s", candidate.name)
        analysis.pattern_candidates_done = position + 1
        session.commit()

    analysis.patterns_extracted_at = utcnow()
    session.commit()
    return _linked_patterns(session, video.id)
