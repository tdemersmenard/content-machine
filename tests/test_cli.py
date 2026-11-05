from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app import cli as cli_module
from app.cli import cli
from app.db.models import IngestJob, JobStatus, Video
from tests.ingest_helpers import FakeClaude, make_deps, make_video

runner = CliRunner()


@pytest.fixture(autouse=True)
def fake_services(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "default_deps", lambda: make_deps())


def test_ingest_file(db: Session, sample_video: Path) -> None:
    result = runner.invoke(cli, ["ingest", str(sample_video), "--own", "--niche", "fitness"])

    assert result.exit_code == 0, result.output
    assert "timeline.json" in result.output
    video = db.scalars(select(Video)).one()
    assert video.is_own_content is True
    assert video.niche_source == "fitness"


def test_ingest_twice_says_nothing_to_do(db: Session, sample_video: Path) -> None:
    runner.invoke(cli, ["ingest", str(sample_video)])

    result = runner.invoke(cli, ["ingest", str(sample_video)])

    assert result.exit_code == 0
    assert "déjà ingérée" in result.output


def test_ingest_no_wait_only_queues(db: Session, sample_video: Path) -> None:
    result = runner.invoke(cli, ["ingest", str(sample_video), "--no-wait"])

    assert result.exit_code == 0
    assert db.scalars(select(IngestJob)).one().status is JobStatus.QUEUED


def test_ingest_missing_file_fails(db: Session) -> None:
    result = runner.invoke(cli, ["ingest", "C:/nexiste/pas.mp4"])

    assert result.exit_code == 1
    assert "introuvable" in result.output


def test_failed_ingest_reports_stage_and_exits_with_error(
    db: Session, sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "default_deps", lambda: make_deps(FakeClaude(fail_after=0)))

    result = runner.invoke(cli, ["ingest", str(sample_video)])

    assert result.exit_code == 1
    assert "vision" in result.output


def test_ingest_folder_takes_videos_only(db: Session, tmp_path: Path) -> None:
    make_video(tmp_path / "a.mp4", "white")
    make_video(tmp_path / "b.MP4", "black")
    (tmp_path / "notes.txt").write_text("ignoré")

    result = runner.invoke(cli, ["ingest-folder", str(tmp_path), "--own"])

    assert result.exit_code == 0, result.output
    assert "2/2" in result.output
    videos = db.scalars(select(Video)).all()
    assert len(videos) == 2
    assert all(video.is_own_content for video in videos)


def test_empty_folder_fails(db: Session, tmp_path: Path) -> None:
    assert runner.invoke(cli, ["ingest-folder", str(tmp_path)]).exit_code == 1


def test_status(db: Session, sample_video: Path) -> None:
    runner.invoke(cli, ["ingest", str(sample_video)])
    video = db.scalars(select(Video)).one()

    result = runner.invoke(cli, ["status", str(video.id)])

    assert "done" in result.output
    assert runner.invoke(cli, ["status", "999"]).exit_code == 1
