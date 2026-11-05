"""Pipeline d'ingestion d'une vidéo.

Chaque étape enregistre son résultat et se saute si elle est déjà faite : un job qui échoue
reprend là où il s'est arrêté, sans refaire la transcription ni repayer les lots de vision.
"""

import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.base import utcnow
from app.db.models import Frame, IngestJob, JobStatus, Shot, SourceType, Transcript, Video
from app.ingest.acquire import DownloadedVideo, download_url
from app.ingest.media import (
    MediaInfo,
    detect_scene_cuts,
    extract_audio,
    extract_frame,
    probe,
    sha256_file,
)
from app.ingest.schemas import Signals, Transcription, Word
from app.ingest.shots import build_shots, clean_cuts, plan_frames
from app.ingest.signals import compute_signals, find_promise
from app.ingest.timeline import build_timeline, write_timeline
from app.ingest.transcribe import Transcriber, transcribe_with_whisper
from app.ingest.vision import analyze_batch
from app.llm.client import ClaudeClient, get_claude


class IngestError(Exception):
    """La vidéo ne peut pas être ingérée."""


@dataclass(frozen=True)
class IngestDeps:
    """Services externes, remplaçables dans les tests."""

    transcriber: Transcriber = transcribe_with_whisper
    claude: Callable[[], ClaudeClient] = get_claude
    downloader: Callable[[str, Path], DownloadedVideo] = download_url


def _root() -> Path:
    return get_settings().data_dir.resolve()


def _relative(path: Path) -> str:
    return path.relative_to(_root()).as_posix()


def _video_dir(file_hash: str) -> Path:
    path = _root() / "videos" / file_hash[:16]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _adopt_file(
    session: Session, job: IngestJob, video: Video, source: Path, keep_source: bool
) -> Video:
    """Range le fichier sous data/videos/<hash>/, ou bascule sur la vidéo qui l'a déjà."""
    file_hash = sha256_file(source)

    existing = session.scalar(
        select(Video).where(Video.file_hash == file_hash, Video.id != video.id)
    )
    if existing is not None:
        video.duplicate_of_id = existing.id
        job.video_id = existing.id
        session.commit()
        if not keep_source:
            source.unlink(missing_ok=True)
        if existing.file_path:
            return existing
        video = existing

    target = _video_dir(file_hash) / f"source{source.suffix.lower()}"
    if keep_source:
        shutil.copy2(source, target)
    else:
        shutil.move(source, target)
    video.file_hash = file_hash
    video.file_path = _relative(target)
    session.commit()
    return video


def _acquire(session: Session, job: IngestJob, video: Video, deps: IngestDeps) -> Video:
    if video.file_path and (_root() / video.file_path).is_file():
        return video

    if job.source_type is not SourceType.URL:
        source = Path(job.source)
        if not source.is_file():
            raise IngestError(f"Fichier introuvable : {source}")
        return _adopt_file(session, job, video, source, job.source_type is SourceType.FILE)

    tmp_root = _root() / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        downloaded = deps.downloader(job.source, Path(tmp))
        target = _adopt_file(session, job, video, downloaded.path, keep_source=False)
    if target.id == video.id:
        for name, value in downloaded.metadata.items():
            setattr(video, name, value)
        session.commit()
    return target


def _check_media(session: Session, video: Video) -> MediaInfo:
    info = probe(_root() / video.file_path)
    if not info.has_video:
        raise IngestError("Le fichier ne contient pas de piste vidéo.")
    if info.duration_s <= 0:
        raise IngestError("Durée de vidéo nulle.")
    limit = get_settings().max_video_duration_s
    if info.duration_s > limit:
        raise IngestError(
            f"Vidéo de {info.duration_s:.0f} s, au-delà de MAX_VIDEO_DURATION_S ({limit:.0f} s)."
        )
    video.duration_s = info.duration_s
    session.commit()
    return info


def _transcribe(session: Session, video: Video, info: MediaInfo, deps: IngestDeps) -> None:
    if session.get(Transcript, video.id) is not None:
        return

    result = Transcription()
    if info.has_audio:
        media = _root() / video.file_path
        audio = media.with_name("audio.wav")
        extract_audio(media, audio)
        try:
            result = deps.transcriber(audio)
        finally:
            audio.unlink(missing_ok=True)

    session.add(
        Transcript(
            video_id=video.id,
            words=[word.model_dump() for word in result.words],
            full_text=result.full_text,
            language=result.language,
        )
    )
    session.commit()


def _cut_shots(session: Session, video: Video) -> None:
    if session.scalar(select(Shot.id).where(Shot.video_id == video.id).limit(1)) is not None:
        return

    media = _root() / video.file_path
    duration_s = video.duration_s
    cuts = clean_cuts(detect_scene_cuts(media, get_settings().scene_threshold), duration_s)

    frames_dir = media.parent / "frames"
    frames_dir.mkdir(exist_ok=True)
    planned = plan_frames(cuts, duration_s)
    frames: list[Frame] = []
    for item in planned:
        path = frames_dir / f"f_{round(item.t_s * 1000):08d}.jpg"
        extract_frame(media, item.t_s, path)
        frames.append(
            Frame(
                video_id=video.id,
                t_s=round(item.t_s, 3),
                reason=item.reason,
                shot_index=item.shot_index,
                frame_path=_relative(path),
            )
        )

    first_frame = {}
    for frame in frames:
        first_frame.setdefault(frame.shot_index, frame.frame_path)
    shots = [
        Shot(
            video_id=video.id,
            index=index,
            start_s=round(start, 3),
            end_s=round(end, 3),
            frame_path=first_frame.get(index),
        )
        for index, (start, end) in enumerate(build_shots(cuts, duration_s))
    ]
    session.add_all([*frames, *shots])
    session.commit()


def _load_frames(session: Session, video: Video) -> list[Frame]:
    return list(
        session.scalars(select(Frame).where(Frame.video_id == video.id).order_by(Frame.t_s))
    )


def _load_shots(session: Session, video: Video) -> list[Shot]:
    return list(session.scalars(select(Shot).where(Shot.video_id == video.id).order_by(Shot.index)))


def _read_frames(session: Session, video: Video, deps: IngestDeps) -> None:
    settings = get_settings()
    pending = [frame for frame in _load_frames(session, video) if frame.analyzed_at is None]

    for offset in range(0, len(pending), settings.vision_batch_size):
        batch = pending[offset : offset + settings.vision_batch_size]
        images = [(frame.t_s, _root() / frame.frame_path) for frame in batch]
        observations = analyze_batch(deps.claude(), images, settings.model_analysis)
        for frame, seen in zip(batch, observations, strict=True):
            frame.on_screen_text = seen.on_screen_text
            frame.visual_description = seen.visual_description
            frame.framing = seen.framing
            frame.face_present = seen.face_present
            frame.camera_motion = seen.camera_motion
            frame.graphic_elements = seen.graphic_elements
            frame.analyzed_at = utcnow()
        session.commit()  # Lot par lot : un lot déjà payé n'est jamais refait.

    frames = _load_frames(session, video)
    for shot in _load_shots(session, video):
        in_shot = [frame for frame in frames if frame.shot_index == shot.index]
        texts = dict.fromkeys(frame.on_screen_text for frame in in_shot if frame.on_screen_text)
        shot.on_screen_text = "\n".join(texts) or None
        shot.visual_description = in_shot[0].visual_description if in_shot else None
    session.commit()


def _compute(session: Session, video: Video, deps: IngestDeps) -> None:
    if video.signals is not None:
        return
    transcript = session.get(Transcript, video.id)
    words = [Word.model_validate(word) for word in transcript.words]
    promise = None
    if words:
        promise = find_promise(deps.claude(), words, get_settings().model_analysis)
    shots = [(shot.start_s, shot.end_s) for shot in _load_shots(session, video)]
    video.signals = compute_signals(words, shots, video.duration_s, promise).model_dump()
    session.commit()


def _write_timeline(session: Session, video: Video) -> None:
    timeline = build_timeline(
        video,
        session.get(Transcript, video.id),
        _load_shots(session, video),
        _load_frames(session, video),
        Signals.model_validate(video.signals),
    )
    path = (_root() / video.file_path).with_name("timeline.json")
    write_timeline(timeline, path)
    video.timeline_path = _relative(path)
    video.ingested_at = utcnow()
    session.commit()


def process_job(session: Session, job: IngestJob, deps: IngestDeps) -> None:
    """Exécute les étapes manquantes. Une exception laisse les étapes finies en place."""

    def enter(stage: str) -> None:
        job.stage = stage
        session.commit()

    video = session.get(Video, job.video_id)

    enter("acquire")
    video = _acquire(session, job, video, deps)

    if video.ingested_at is None:
        info = _check_media(session, video)
        enter("audio")
        _transcribe(session, video, info, deps)
        enter("shots")
        _cut_shots(session, video)
        enter("vision")
        _read_frames(session, video, deps)
        enter("signals")
        _compute(session, video, deps)
        enter("timeline")
        _write_timeline(session, video)

    job.status = JobStatus.DONE
    job.stage = None
    job.finished_at = utcnow()
    session.commit()
