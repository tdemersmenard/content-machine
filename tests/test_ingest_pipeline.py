"""Pipeline complet : vrai ffmpeg, Whisper et Claude simulés."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.base import utcnow
from app.db.models import Frame, IngestJob, JobStatus, SourceType, Video
from app.ingest.jobs import drain, ingest, requeue_stale_jobs, run_job, run_next, submit
from app.ingest.pipeline import IngestError
from app.ingest.vision import _batch_schema
from tests.ingest_helpers import FakeClaude, FakeTranscriber, make_deps, make_video


def data_path(relative: str) -> Path:
    return get_settings().data_dir.resolve() / relative


def count(db: Session, model: type) -> int:
    return db.scalar(select(func.count()).select_from(model))


def test_full_ingestion_produces_a_complete_bundle(db: Session, sample_video: Path) -> None:
    bundle = ingest(
        str(sample_video), is_own_content=True, niche_source="fitness", deps=make_deps()
    )

    assert bundle.video.is_own_content is True
    assert bundle.video.niche_source == "fitness"
    assert bundle.video.duration_s == pytest.approx(9, abs=0.2)
    assert bundle.video.ingested_at is not None
    assert bundle.video.views is None  # Fichier local : aucune stat n'est inventée.
    assert data_path(bundle.video.file_path).is_file()
    assert sample_video.is_file()  # L'original de l'utilisateur reste en place.

    assert bundle.transcript.language == "en"
    assert bundle.transcript.words[0]["start"] == 0.5

    assert [(s.start_s, s.index) for s in bundle.shots][0] == (0.0, 0)
    assert len(bundle.shots) == 3
    assert bundle.shots[1].start_s == pytest.approx(3.0, abs=0.1)
    assert bundle.shots[1].on_screen_text == "DOUBLE TES VUES"
    assert all(data_path(shot.frame_path).is_file() for shot in bundle.shots)

    times = [frame.t_s for frame in bundle.frames]
    assert times[:3] == [0.0, 1.0, 2.0]
    assert all(frame.framing == "graphic" for frame in bundle.frames)
    assert all(data_path(frame.frame_path).is_file() for frame in bundle.frames)

    signals = bundle.signals
    assert signals.shot_count == 3
    assert signals.cuts_first_3s in (0, 1)  # Le cut tombe à 3 s, à la frontière.
    assert signals.words_per_second == pytest.approx(10 / 9, abs=0.05)
    assert signals.promise.text == "double your views"
    assert signals.promise.start_s == pytest.approx(2.6)
    assert signals.promise.end_s == pytest.approx(3.45)


def test_timeline_merges_everything_on_one_time_axis(db: Session, sample_video: Path) -> None:
    bundle = ingest(str(sample_video), deps=make_deps())

    timeline = json.loads(data_path(bundle.timeline_path).read_text("utf-8"))

    assert timeline["version"] == 1
    assert timeline["video"]["stats"]["views"] is None
    assert timeline["video"]["language"] == "en"
    assert timeline["signals"]["shot_count"] == 3
    starts = [event["start_s"] for event in timeline["events"]]
    assert starts == sorted(starts)
    assert {event["type"] for event in timeline["events"]} == {
        "shot",
        "frame",
        "speech",
        "silence",
        "promise",
    }
    speech = next(event for event in timeline["events"] if event["type"] == "speech")
    assert speech["words"][0] == {"word": "Today", "start": 0.5, "end": 0.75}
    frame = next(event for event in timeline["events"] if event["type"] == "frame")
    assert frame["on_screen_text"] == "DOUBLE TES VUES"


def test_same_file_twice_does_nothing_the_second_time(db: Session, sample_video: Path) -> None:
    claude = FakeClaude()
    transcriber = FakeTranscriber()
    deps = make_deps(claude, transcriber)
    first = ingest(str(sample_video), deps=deps)
    calls = len(claude.calls)

    second = ingest(str(sample_video), deps=deps)

    assert second.video.id == first.video.id
    assert transcriber.calls == 1
    assert len(claude.calls) == calls
    assert count(db, Video) == 1
    assert count(db, IngestJob) == 1


def test_same_file_under_another_name_is_detected(
    db: Session, sample_video: Path, tmp_path: Path
) -> None:
    copy = tmp_path / "copie.mp4"
    copy.write_bytes(sample_video.read_bytes())
    deps = make_deps()

    assert ingest(str(copy), deps=deps).video.id == ingest(str(sample_video), deps=deps).video.id


def test_same_url_twice_does_not_download_again(db: Session, sample_video: Path) -> None:
    url = "https://example.com/v/1"
    deps = make_deps(downloads={url: sample_video}, metadata={"creator": "Ada", "views": 12})

    first = ingest(url, deps=deps)
    second = ingest(url, deps=make_deps())  # Sans table de téléchargement : planterait.

    assert first.video.creator == "Ada"
    assert first.video.views == 12
    assert first.video.likes is None
    assert second.video.id == first.video.id


def test_two_urls_with_the_same_file_share_one_video(db: Session, sample_video: Path) -> None:
    urls = {"https://a.example/1": sample_video, "https://b.example/2": sample_video}
    transcriber = FakeTranscriber()
    deps = make_deps(transcriber=transcriber, downloads=urls)

    first = ingest("https://a.example/1", deps=deps)
    second = ingest("https://b.example/2", deps=deps)

    assert second.video.id == first.video.id
    assert transcriber.calls == 1
    duplicate = db.scalar(select(Video).where(Video.source_url == "https://b.example/2"))
    assert duplicate.duplicate_of_id == first.video.id


def test_failure_resumes_without_redoing_paid_work(
    db: Session, sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VISION_BATCH_SIZE", "3")
    get_settings.cache_clear()
    transcriber = FakeTranscriber()
    failing = FakeClaude(fail_after=1)  # Le 1er lot passe, le 2e tombe.

    with pytest.raises(IngestError, match="Panne simulée"):
        ingest(str(sample_video), deps=make_deps(failing, transcriber))

    job = db.scalars(select(IngestJob)).one()
    assert (job.status, job.stage) == (JobStatus.FAILED, "vision")
    assert "Panne simulée" in job.error
    analyzed = db.scalar(
        select(func.count()).select_from(Frame).where(Frame.analyzed_at.is_not(None))
    )
    total = count(db, Frame)
    assert analyzed == 3

    recovered = FakeClaude()
    bundle = ingest(str(sample_video), deps=make_deps(recovered, transcriber))

    assert bundle.video.ingested_at is not None
    assert transcriber.calls == 1
    sent = [c for c in recovered.calls if c["prompt"] == "frames"]
    assert sum(len(c["images"]) for c in sent) == total - 3


def test_video_without_audio_has_an_empty_transcript_and_no_promise(
    db: Session, silent_video: Path
) -> None:
    claude = FakeClaude()
    transcriber = FakeTranscriber()

    bundle = ingest(str(silent_video), deps=make_deps(claude, transcriber))

    assert transcriber.calls == 0
    assert claude.count("promise") == 0
    assert bundle.transcript.words == []
    assert bundle.transcript.language is None
    assert bundle.signals.words_per_second == 0
    assert bundle.signals.promise is None


def test_promise_with_out_of_range_indices_is_dropped(db: Session, sample_video: Path) -> None:
    class WrongIndices(FakeClaude):
        def call_json(self, *, prompt, version, schema, model, **kwargs):
            result = super().call_json(
                prompt=prompt, version=version, schema=schema, model=model, **kwargs
            )
            if prompt == "promise":
                result.data.end_word = 999
            return result

    bundle = ingest(str(sample_video), deps=make_deps(WrongIndices()))

    assert bundle.signals.promise is None


def test_video_over_the_duration_limit_is_refused_before_any_paid_work(
    db: Session, sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_VIDEO_DURATION_S", "5")
    get_settings.cache_clear()
    claude = FakeClaude()
    transcriber = FakeTranscriber()

    with pytest.raises(IngestError, match="MAX_VIDEO_DURATION_S"):
        ingest(str(sample_video), deps=make_deps(claude, transcriber))

    assert transcriber.calls == 0
    assert claude.calls == []


def test_invalid_file_fails_cleanly(db: Session, tmp_path: Path) -> None:
    fake = tmp_path / "fake.mp4"
    fake.write_text("pas une vidéo")

    with pytest.raises(IngestError):
        ingest(str(fake), deps=make_deps())

    assert db.scalars(select(IngestJob)).one().status is JobStatus.FAILED


def test_missing_file_is_refused_at_submission(db: Session) -> None:
    with pytest.raises(IngestError, match="introuvable"):
        submit(db, "C:/nexiste/pas.mp4")


def test_queue_processes_jobs_in_order(db: Session, sample_video: Path, tmp_path: Path) -> None:
    other = make_video(tmp_path / "other.mp4", "white")
    first = submit(db, str(sample_video))
    second = submit(db, str(other))
    deps = make_deps()

    assert db.get(IngestJob, first.job_id).status is JobStatus.QUEUED
    assert run_next(deps) is True
    db.expire_all()
    assert db.get(IngestJob, first.job_id).status is JobStatus.DONE
    assert db.get(IngestJob, second.job_id).status is JobStatus.QUEUED

    drain(deps)
    db.expire_all()
    assert db.get(IngestJob, second.job_id).status is JobStatus.DONE
    assert run_next(deps) is False


def test_submitting_a_queued_video_again_does_not_add_a_job(
    db: Session, sample_video: Path
) -> None:
    first = submit(db, str(sample_video))
    second = submit(db, str(sample_video))

    assert second.job_id == first.job_id
    assert count(db, IngestJob) == 1


def test_a_job_cannot_run_twice(db: Session, sample_video: Path) -> None:
    transcriber = FakeTranscriber()
    deps = make_deps(transcriber=transcriber)
    job_id = submit(db, str(sample_video)).job_id

    run_job(job_id, deps)
    run_job(job_id, deps)

    assert transcriber.calls == 1


def test_upload_is_moved_and_removed_when_duplicate(
    db: Session, sample_video: Path, tmp_path: Path
):
    deps = make_deps()
    upload = tmp_path / "upload.mp4"
    upload.write_bytes(sample_video.read_bytes())
    submission = submit(db, str(upload), source_type=SourceType.UPLOAD)
    run_job(submission.job_id, deps)
    assert not upload.exists()

    upload.write_bytes(sample_video.read_bytes())
    again = submit(db, str(upload), source_type=SourceType.UPLOAD)

    assert again.deduplicated is True
    assert not upload.exists()


def test_stale_running_job_is_requeued(db: Session, sample_video: Path) -> None:
    job = db.get(IngestJob, submit(db, str(sample_video)).job_id)
    job.status = JobStatus.RUNNING
    job.started_at = utcnow() - timedelta(hours=3)
    db.commit()

    assert requeue_stale_jobs(db) == 1
    db.refresh(job)
    assert job.status is JobStatus.QUEUED


def test_recent_running_job_is_left_alone(db: Session, sample_video: Path) -> None:
    job = db.get(IngestJob, submit(db, str(sample_video)).job_id)
    job.status = JobStatus.RUNNING
    job.started_at = utcnow()
    db.commit()

    assert requeue_stale_jobs(db) == 0


def test_vision_batch_must_describe_every_image() -> None:
    observation = {
        "on_screen_text": "",
        "visual_description": "x",
        "framing": "other",
        "face_present": False,
        "camera_motion": "unknown",
        "graphic_elements": [],
    }
    schema = _batch_schema(2)

    assert schema(
        observations=[{"image_number": 2, **observation}, {"image_number": 1, **observation}]
    )
    with pytest.raises(ValueError, match="une observation par image"):
        schema(observations=[{"image_number": 1, **observation}])


def test_frames_prompt_lists_the_timestamps() -> None:
    from app.llm.prompts import load_prompt

    text = load_prompt("frames", "v1").render_user({"timestamps": [0.0, 1.5]})

    assert "Image 1 : 0.00 s" in text
    assert "Image 2 : 1.50 s" in text


def test_promise_prompt_numbers_the_words() -> None:
    from app.llm.prompts import load_prompt

    text = load_prompt("promise", "v1").render_user({"words": ["Salut", "toi"]})

    assert "[0]Salut [1]toi" in text
