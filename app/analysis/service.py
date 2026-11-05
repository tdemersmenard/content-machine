"""Analyse virale d'une vidéo ingérée.

Passe 1 : Claude analyse la timeline, les métadonnées et les frames du hook.
Passe 2 : Claude relit l'analyse contre la timeline et retire ou corrige ce qui n'est pas prouvé.
Enfin le code contrôle chaque preuve : c'est ce contrôle qui a le dernier mot.
"""

import json
import logging
from pathlib import Path

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.analysis.context import build_metadata, compute_metrics
from app.analysis.evidence import audit, build_index
from app.analysis.report import render_report
from app.analysis.schemas import (
    AnalysisDocument,
    DraftAnalysis,
    Verification,
    VerifiedAnalysis,
    ViralAnalysis,
)
from app.config import get_settings
from app.db.models import Analysis, Frame, Video
from app.llm.client import ClaudeClient, get_claude
from app.llm.prompts import load_prompt
from app.patterns.embeddings import Embedder
from app.patterns.service import extract_patterns

logger = logging.getLogger(__name__)

PROMPT_NAME = "analysis"
PROMPT_VERSION = "v1"
VERIFY_PROMPT_NAME = "analysis_verify"
VERIFY_PROMPT_VERSION = "v1"
CURRENT_PROMPT_VERSION = f"{PROMPT_NAME}/{PROMPT_VERSION}"
HOOK_FRAMES_END_S = 5.0


class AnalysisError(Exception):
    """La vidéo ne peut pas être analysée."""


def _root() -> Path:
    return get_settings().data_dir.resolve()


def _extract_patterns_safely(
    session: Session, analysis: Analysis, claude: ClaudeClient, embedder: Embedder | None
) -> None:
    """Un échec ici ne doit pas faire perdre une analyse déjà payée : on réessaiera plus tard."""
    try:
        extract_patterns(session, analysis.id, claude, embedder)
    except Exception:
        logger.exception("Extraction des patterns échouée pour l'analyse %d", analysis.id)
        session.rollback()


def report_path(video: Video, analysis_id: int) -> Path:
    return (_root() / video.timeline_path).with_name(f"analysis_{analysis_id}.md")


def _draft_path(video: Video, prompt_hash: str) -> Path:
    return (_root() / video.timeline_path).with_name(f"analysis_draft_{prompt_hash}.json")


def _run_draft(
    claude: ClaudeClient,
    video: Video,
    session: Session,
    timeline: str,
    metadata: str,
    force: bool,
) -> DraftAnalysis:
    """Passe 1. Le brouillon est gardé sur disque : si la passe 2 échoue, il n'est pas repayé."""
    path = _draft_path(video, load_prompt(PROMPT_NAME, PROMPT_VERSION).content_hash)
    if path.is_file() and not force:
        return DraftAnalysis.model_validate_json(path.read_text("utf-8"))

    frames = session.scalars(
        select(Frame)
        .where(Frame.video_id == video.id, Frame.t_s <= HOOK_FRAMES_END_S)
        .order_by(Frame.t_s)
    ).all()
    result = claude.call_json(
        prompt=PROMPT_NAME,
        version=PROMPT_VERSION,
        schema=DraftAnalysis,
        model=get_settings().model_analysis,
        variables={
            "timeline": timeline,
            "metadata": metadata,
            "timestamps": [frame.t_s for frame in frames],
        },
        images=[_root() / frame.frame_path for frame in frames],
        effort="high",
        constrained=False,  # Schéma trop gros pour la contrainte de l'API.
    )
    path.write_text(result.data.model_dump_json(), encoding="utf-8")
    return result.data


def analyze(
    session: Session,
    video_id: int,
    claude: ClaudeClient | None = None,
    force: bool = False,
    embedder: Embedder | None = None,
) -> Analysis:
    """Analyse une vidéo et renvoie la ligne `analyses`.

    Si une analyse existe déjà avec la version de prompt courante, elle est renvoyée sans appel
    à Claude, sauf avec `force`.
    """
    video = session.get(Video, video_id)
    if video is None:
        raise LookupError(f"Vidéo {video_id} inconnue.")
    while video.duplicate_of_id is not None:
        video = session.get(Video, video.duplicate_of_id)
    if video.ingested_at is None or video.timeline_path is None:
        raise AnalysisError(f"Vidéo {video.id} : ingestion non terminée.")

    if not force:
        existing = session.scalar(
            select(Analysis)
            .where(Analysis.video_id == video.id, Analysis.prompt_version == CURRENT_PROMPT_VERSION)
            .order_by(Analysis.id.desc())
            .limit(1)
        )
        if existing is not None:
            if existing.patterns_extracted_at is None:
                _extract_patterns_safely(session, existing, claude or get_claude(), embedder)
            return existing

    claude = claude or get_claude()
    settings = get_settings()
    timeline = json.loads((_root() / video.timeline_path).read_text("utf-8"))
    metrics = compute_metrics(video)
    metadata = build_metadata(video, metrics)
    timeline_json = json.dumps(timeline, ensure_ascii=False)
    metadata_json = json.dumps(metadata, ensure_ascii=False)

    draft = _run_draft(claude, video, session, timeline_json, metadata_json, force)

    # Les erreurs déjà visibles par le code sont signalées à la passe 2 plutôt que découvertes.
    index = build_index(timeline, metadata)
    precheck = audit(ViralAnalysis.model_validate(draft.model_dump()), index)

    verified = claude.call_json(
        prompt=VERIFY_PROMPT_NAME,
        version=VERIFY_PROMPT_VERSION,
        schema=VerifiedAnalysis,
        model=settings.model_analysis,
        variables={
            "timeline": timeline_json,
            "metadata": metadata_json,
            "analysis": draft.model_dump_json(),
            "code_findings": json.dumps(
                [item.model_dump() for item in precheck.removals], ensure_ascii=False
            ),
        },
        effort="high",
        constrained=False,  # Schéma trop gros pour la contrainte de l'API.
    )

    final = verified.data.analysis
    code_check = audit(final, index)

    document = AnalysisDocument(
        analysis=final,
        metrics=metrics,
        verification=Verification(
            prompt_version=verified.prompt_version,
            model=verified.model,
            llm_corrections=verified.data.corrections,
            llm_removals=verified.data.removals,
            code_corrections=code_check.corrections,
            code_removals=code_check.removals,
        ),
    )

    analysis = Analysis(
        video_id=video.id,
        prompt_version=CURRENT_PROMPT_VERSION,
        model=verified.model,
        json_result=document.model_dump(mode="json"),
    )
    session.add(analysis)
    session.commit()

    report_path(video, analysis.id).write_text(render_report(document), encoding="utf-8")
    _draft_path(video, load_prompt(PROMPT_NAME, PROMPT_VERSION).content_hash).unlink(
        missing_ok=True
    )
    _extract_patterns_safely(session, analysis, claude, embedder)
    return analysis


def pending_video_ids(session: Session) -> list[int]:
    """Vidéos ingérées sans analyse courante, ou dont les patterns n'ont pas été extraits."""
    analyzed = exists().where(
        Analysis.video_id == Video.id,
        Analysis.prompt_version == CURRENT_PROMPT_VERSION,
        Analysis.patterns_extracted_at.is_not(None),
    )
    return list(
        session.scalars(
            select(Video.id)
            .where(Video.ingested_at.is_not(None), Video.duplicate_of_id.is_(None), ~analyzed)
            .order_by(Video.id)
        )
    )
