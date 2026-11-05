"""File de jobs en base et fonction `ingest`.

Pas de broker : un worker interroge la table ingest_jobs. La prise d'un job est atomique
(UPDATE conditionnel), donc l'API et la CLI peuvent travailler en même temps.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.db.models import Frame, IngestJob, JobStatus, Shot, SourceType, Transcript, Video
from app.db.session import get_session_factory
from app.ingest.acquire import AcquireError
from app.ingest.media import MediaError, sha256_file
from app.ingest.pipeline import IngestDeps, IngestError, process_job
from app.ingest.schemas import (
    FrameOut,
    ShotOut,
    Signals,
    TranscriptOut,
    VideoBundle,
    VideoOut,
)
from app.llm.errors import ClaudeError

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 2.0
STALE_AFTER = timedelta(hours=2)  # Un job « running » aussi vieux a perdu son worker.
ACTIVE = (JobStatus.QUEUED, JobStatus.RUNNING)
# Échecs normaux, stockés sur le job sans traceback : fichier invalide, clé absente, etc.
EXPECTED_ERRORS = (IngestError, MediaError, AcquireError, ClaudeError)


@dataclass(frozen=True)
class Submission:
    video_id: int
    job_id: int | None
    deduplicated: bool  # Vrai si rien n'a été mis en file : la vidéo est déjà ingérée.


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def canonical(session: Session, video: Video) -> Video:
    """Suit le lien vers la vidéo de référence si celle-ci est un doublon."""
    while video.duplicate_of_id is not None:
        video = session.get(Video, video.duplicate_of_id)
    return video


def latest_job(session: Session, video_id: int) -> IngestJob | None:
    return session.scalar(
        select(IngestJob)
        .where(IngestJob.video_id == video_id)
        .order_by(IngestJob.id.desc())
        .limit(1)
    )


def submit(
    session: Session,
    source: str,
    *,
    source_type: SourceType | None = None,
    is_own_content: bool = False,
    niche_source: str | None = None,
) -> Submission:
    """Enregistre une source. Si la vidéo est déjà connue, ne relance que ce qui manque."""
    if source_type is None:
        source_type = SourceType.URL if is_url(source) else SourceType.FILE

    if source_type is SourceType.URL:
        file_hash = None
        existing = session.scalar(select(Video).where(Video.source_url == source))
    else:
        path = Path(source)
        if not path.is_file():
            raise IngestError(f"Fichier introuvable : {path}")
        file_hash = sha256_file(path)
        existing = session.scalar(select(Video).where(Video.file_hash == file_hash))

    if existing is not None:
        video = canonical(session, existing)
        job = latest_job(session, video.id)
        reusable = video.ingested_at is not None or (job is not None and job.status in ACTIVE)
        if source_type is SourceType.UPLOAD and (reusable or video.file_path):
            Path(source).unlink(missing_ok=True)
        if video.ingested_at is not None:
            return Submission(video.id, job.id if job else None, deduplicated=True)
        if job is not None and job.status in ACTIVE:
            return Submission(video.id, job.id, deduplicated=False)
    else:
        video = Video(
            source_url=source if source_type is SourceType.URL else None,
            file_hash=file_hash,
            is_own_content=is_own_content,
            niche_source=niche_source,
        )
        session.add(video)
        session.flush()

    job = IngestJob(video_id=video.id, source=source, source_type=source_type)
    session.add(job)
    session.commit()
    return Submission(video.id, job.id, deduplicated=False)


def _claim(session: Session, job_id: int) -> bool:
    result = session.execute(
        update(IngestJob)
        .where(IngestJob.id == job_id, IngestJob.status == JobStatus.QUEUED)
        .values(status=JobStatus.RUNNING, started_at=utcnow(), error=None)
    )
    session.commit()
    return result.rowcount == 1


def run_job(job_id: int, deps: IngestDeps | None = None) -> None:
    """Exécute un job en attente. Une erreur est stockée sur le job, jamais propagée."""
    with get_session_factory()() as session:
        if not _claim(session, job_id):
            return
        job = session.get(IngestJob, job_id)
        try:
            process_job(session, job, deps or IngestDeps())
        except Exception as exc:
            if isinstance(exc, EXPECTED_ERRORS):
                logger.info("Job d'ingestion %d en échec : %s", job_id, exc)
            else:
                logger.exception("Erreur inattendue du job d'ingestion %d", job_id)
            session.rollback()
            job = session.get(IngestJob, job_id)
            job.status = JobStatus.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            job.finished_at = utcnow()
            session.commit()


def requeue_stale_jobs(session: Session) -> int:
    result = session.execute(
        update(IngestJob)
        .where(
            IngestJob.status == JobStatus.RUNNING,
            IngestJob.started_at < utcnow() - STALE_AFTER,
        )
        .values(status=JobStatus.QUEUED)
    )
    session.commit()
    return result.rowcount


def run_next(deps: IngestDeps | None = None) -> bool:
    """Exécute le plus ancien job en attente. Renvoie False si la file est vide."""
    with get_session_factory()() as session:
        job_id = session.scalar(
            select(IngestJob.id)
            .where(IngestJob.status == JobStatus.QUEUED)
            .order_by(IngestJob.id)
            .limit(1)
        )
    if job_id is None:
        return False
    run_job(job_id, deps)
    return True


def drain(deps: IngestDeps | None = None) -> None:
    """Vide la file, puis rend la main."""
    with get_session_factory()() as session:
        requeue_stale_jobs(session)
    while run_next(deps):
        pass


class Worker(threading.Thread):
    """Traite la file en tâche de fond, tant que `stop` n'a pas été appelé."""

    def __init__(self, deps: IngestDeps | None = None) -> None:
        super().__init__(name="ingest-worker", daemon=True)
        self._deps = deps
        self._stopping = threading.Event()

    def stop(self) -> None:
        self._stopping.set()

    def run(self) -> None:
        while not self._stopping.is_set():
            try:
                drain(self._deps)
            except Exception:
                logger.exception("Erreur du worker d'ingestion")
            self._stopping.wait(POLL_INTERVAL_S)


def load_bundle(session: Session, video: Video) -> VideoBundle:
    transcript = session.get(Transcript, video.id)
    shots = session.scalars(select(Shot).where(Shot.video_id == video.id).order_by(Shot.index))
    frames = session.scalars(select(Frame).where(Frame.video_id == video.id).order_by(Frame.t_s))
    return VideoBundle(
        video=VideoOut.model_validate(video),
        transcript=TranscriptOut.model_validate(transcript) if transcript else None,
        shots=[ShotOut.model_validate(shot) for shot in shots],
        frames=[FrameOut.model_validate(frame) for frame in frames],
        signals=Signals.model_validate(video.signals) if video.signals else None,
        timeline_path=video.timeline_path,
    )


def ingest(
    source: str,
    *,
    is_own_content: bool = False,
    niche_source: str | None = None,
    deps: IngestDeps | None = None,
) -> VideoBundle:
    """Ingère un fichier local ou une URL et renvoie le bundle complet.

    Synchrone. Lève IngestError si le pipeline échoue. La reprise est automatique au
    prochain appel, à partir de la dernière étape terminée.
    """
    with get_session_factory()() as session:
        submission = submit(
            session, source, is_own_content=is_own_content, niche_source=niche_source
        )

    if submission.job_id is not None and not submission.deduplicated:
        run_job(submission.job_id, deps)

    with get_session_factory()() as session:
        job = session.get(IngestJob, submission.job_id) if submission.job_id else None
        if job is not None and job.status is not JobStatus.DONE:
            raise IngestError(f"Ingestion non terminée ({job.status}) : {job.error}")
        video_id = job.video_id if job else submission.video_id
        return load_bundle(session, session.get(Video, video_id))
