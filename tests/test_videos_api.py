import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import get_settings
from app.ingest import jobs
from app.ingest.jobs import drain
from app.main import app
from tests.ingest_helpers import make_deps

client = TestClient(app)


def upload(path: Path, **form: str):
    with path.open("rb") as stream:
        return client.post("/videos", files={"file": ("ma vidéo.MP4", stream)}, data=form)


def test_upload_then_status_then_bundle(db: Session, sample_video: Path) -> None:
    created = upload(sample_video, is_own_content="true", niche_source="fitness")

    assert created.status_code == 202
    body = created.json()
    assert body["status"] == "queued"
    assert body["deduplicated"] is False

    pending = client.get(f"/videos/{body['video_id']}").json()
    assert pending["status"] == "queued"
    assert pending["bundle"] is None

    drain(make_deps())

    done = client.get(f"/videos/{body['video_id']}").json()
    assert done["status"] == "done"
    assert done["bundle"]["video"]["is_own_content"] is True
    assert done["bundle"]["video"]["niche_source"] == "fitness"
    assert done["bundle"]["signals"]["shot_count"] == 3
    assert done["bundle"]["timeline_path"].endswith("timeline.json")


def test_uploading_the_same_file_again_returns_the_existing_video(
    db: Session, sample_video: Path
) -> None:
    video_id = upload(sample_video).json()["video_id"]
    drain(make_deps())

    again = upload(sample_video)

    assert again.status_code == 200
    assert again.json()["video_id"] == video_id
    assert again.json()["deduplicated"] is True
    assert again.json()["status"] == "done"
    uploads = get_settings().data_dir.resolve() / "uploads"
    assert list(uploads.iterdir()) == []


def test_url_submission(db: Session, sample_video: Path) -> None:
    url = "https://example.com/v/1"

    created = client.post("/videos", data={"url": url})

    assert created.status_code == 202
    drain(make_deps(downloads={url: sample_video}, metadata={"creator": "Ada"}))
    body = client.get(f"/videos/{created.json()['video_id']}").json()
    assert body["bundle"]["video"]["source_url"] == url
    assert body["bundle"]["video"]["creator"] == "Ada"


def test_duplicate_url_placeholder_shows_the_reference_video(
    db: Session, sample_video: Path
) -> None:
    urls = {"https://a.example/1": sample_video, "https://b.example/2": sample_video}
    first = client.post("/videos", data={"url": "https://a.example/1"}).json()
    second = client.post("/videos", data={"url": "https://b.example/2"}).json()

    drain(make_deps(downloads=urls))

    body = client.get(f"/videos/{second['video_id']}").json()
    assert body["id"] == first["video_id"]
    assert body["status"] == "done"
    assert body["bundle"] is not None


def test_failed_job_exposes_stage_and_error(db: Session, tmp_path: Path) -> None:
    fake = tmp_path / "fake.mp4"
    fake.write_text("pas une vidéo")
    video_id = upload(fake).json()["video_id"]

    drain(make_deps())

    body = client.get(f"/videos/{video_id}").json()
    assert body["status"] == "failed"
    assert body["stage"] == "acquire"
    assert "MediaError" in body["error"]
    assert body["bundle"] is None


@pytest.mark.parametrize(
    "form",
    [
        {},
        {"url": "file:///C:/Windows/win.ini"},
        {"url": "C:/video.mp4"},
    ],
)
def test_invalid_submissions_are_rejected(db: Session, form: dict[str, str]) -> None:
    assert client.post("/videos", data=form).status_code == 422


def test_file_and_url_together_are_rejected(db: Session, sample_video: Path) -> None:
    assert upload(sample_video, url="https://example.com/1").status_code == 422


def test_unknown_video_returns_404(db: Session) -> None:
    assert client.get("/videos/999").status_code == 404


def test_background_worker_processes_the_queue(
    db: Session, sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs, "POLL_INTERVAL_S", 0.05)
    monkeypatch.setattr(jobs, "IngestDeps", lambda: make_deps())

    with TestClient(app) as running:
        with sample_video.open("rb") as stream:
            created = running.post("/videos", files={"file": ("v.mp4", stream)}).json()

        deadline = time.monotonic() + 30
        status = None
        while time.monotonic() < deadline:
            status = running.get(f"/videos/{created['video_id']}").json()["status"]
            if status in ("done", "failed"):
                break
            time.sleep(0.05)

    assert status == "done"
