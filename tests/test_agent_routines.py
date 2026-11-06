"""Routines planifiées : quotidien, rappels, rapport du lundi et dossier surveillé."""

import shutil
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app import cli as cli_module
from app.agent import routines
from app.agent.budget import paid_work
from app.agent.routines import (
    Inbox,
    Services,
    daily_due,
    inbox_root,
    run_daily,
    run_reminders,
    run_weekly,
    week_start,
    weekly_due,
)
from app.agent.scheduler import build_scheduler
from app.cli import cli
from app.config import get_settings
from app.db.models import (
    ActionStatus,
    AgentAction,
    Analysis,
    IngestJob,
    JobStatus,
    Niche,
    Notification,
    PostMortem,
    Script,
    Video,
    WeeklyReport,
)
from app.generate.service import load_content
from app.ingest.jobs import submit
from app.learning.service import save_performance
from app.main import app
from tests.agent_helpers import FakeWeeklyClaude, spend
from tests.analysis_helpers import FakeAnalysisClaude
from tests.conftest import distinct_vectors
from tests.ingest_helpers import make_deps
from tests.script_helpers import script_draft
from tests.test_learning import add_pattern, add_script, history, stats
from tests.test_niche import make_niche

# Lundi 2 novembre 2026, 8 h à Montréal.
MONDAY = datetime(2026, 11, 2, 13, 0, tzinfo=UTC)


@pytest.fixture
def niche(db: Session) -> Niche:
    return make_niche(db)


def services(claude=None, deps=None) -> Services:
    return Services(claude or FakeAnalysisClaude(), deps or make_deps(), distinct_vectors)


def small_budget(monkeypatch: pytest.MonkeyPatch, tokens: int) -> None:
    monkeypatch.setenv("AGENT_DAILY_TOKEN_BUDGET", str(tokens))
    get_settings.cache_clear()


def notifications(db: Session, kind: str) -> list[Notification]:
    db.expire_all()
    return list(db.scalars(select(Notification).where(Notification.kind == kind)))


def test_daily_is_due_once_after_its_hour(db: Session) -> None:
    before = datetime(2026, 11, 6, 10, 0, tzinfo=UTC)  # 5 h à Montréal.
    after = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)  # 7 h à Montréal.

    assert not daily_due(db, before)
    assert daily_due(db, after)

    db.add(
        AgentAction(
            source="routine",
            action="routine_daily",
            status=ActionStatus.OK,
            created_at=after,
        )
    )
    db.commit()
    assert not daily_due(db, after + timedelta(hours=2))
    assert daily_due(db, after + timedelta(days=1))


def test_failed_daily_is_still_due(db: Session) -> None:
    after = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)
    db.add(
        AgentAction(
            source="routine", action="routine_daily", status=ActionStatus.ERROR, created_at=after
        )
    )
    db.commit()

    assert daily_due(db, after)


def test_daily_processes_the_queue_then_analyzes(db: Session, sample_video: Path) -> None:
    submit(db, str(sample_video))
    fake = FakeAnalysisClaude()

    result = run_daily(services(fake), force=True)

    assert result["ingested"] == 1
    assert len(result["analyzed"]) == 1
    db.expire_all()
    assert db.scalar(select(func.count()).select_from(Analysis)) == 1
    assert db.scalar(select(IngestJob.status)) is JobStatus.DONE


def test_daily_skips_when_not_due(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routines, "daily_due", lambda *_: False)

    assert run_daily(services()) is None
    assert db.scalar(select(func.count()).select_from(AgentAction)) == 0


def test_daily_stops_at_the_budget_and_warns_once(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    db.add(Video(file_hash="a", ingested_at=MONDAY, timeline_path="x"))
    db.commit()
    small_budget(monkeypatch, 60_000)
    fake = FakeAnalysisClaude()

    first = run_daily(services(fake), force=True)
    second = run_daily(services(fake), force=True)

    assert first["analyzed"] == second["analyzed"] == []
    assert len(notifications(db, "budget")) == 1


def test_one_failed_analysis_does_not_block_the_others(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    db.add_all(Video(file_hash=name, ingested_at=MONDAY, timeline_path="x") for name in ("a", "b"))
    db.commit()
    seen = []

    def fake_analyze(session, video_id, *_, **__):
        seen.append(video_id)
        if len(seen) == 1:
            raise LookupError("Vidéo cassée")

    monkeypatch.setattr(routines, "analyze", fake_analyze)

    result = run_daily(services(), force=True)

    assert len(result["analyzed"]) == 1
    assert len(result["failed"]) == 1


def published(db: Session, niche: Niche, hours_ago: float) -> Script:
    script = add_script(db, niche, [])
    script.published_at = datetime.now(UTC) - timedelta(hours=hours_ago)
    db.commit()
    return script


def test_reminders_fire_at_each_horizon_only_once(db: Session, niche: Niche) -> None:
    script = published(db, niche, 80)

    assert run_reminders() == 2
    assert run_reminders() == 0
    assert run_reminders(datetime.now(UTC) + timedelta(days=5)) == 1

    rows = notifications(db, "stats_reminder")
    assert sorted(item.horizon_h for item in rows) == [24, 72, 168]
    assert all(item.script_id == script.id for item in rows)
    assert f"cm stats {script.id}" in rows[0].body


def test_no_reminder_before_the_horizon_or_for_unpublished_scripts(
    db: Session, niche: Niche
) -> None:
    published(db, niche, 23)
    add_script(db, niche, [])

    assert run_reminders() == 0


def test_entered_stats_skip_and_close_reminders(db: Session, niche: Niche) -> None:
    script = published(db, niche, 30)
    run_reminders()
    [reminder] = notifications(db, "stats_reminder")
    assert reminder.read_at is None

    save_performance(db, script, stats(horizon=24))
    assert run_reminders() == 0

    [reminder] = notifications(db, "stats_reminder")
    assert reminder.read_at is not None


def test_reminders_are_logged(db: Session, niche: Niche) -> None:
    published(db, niche, 30)

    run_reminders()

    assert db.scalar(select(AgentAction.action)) == "stats_reminder"


def test_week_starts_on_monday_at_the_routine_hour() -> None:
    wednesday = datetime(2026, 11, 4, 15, 0, tzinfo=UTC)

    assert week_start(wednesday).isoformat() == "2026-11-02T07:00:00-05:00"


def test_weekly_is_due_from_monday_morning_until_it_ran(db: Session, niche: Niche) -> None:
    early = datetime(2026, 11, 2, 11, 0, tzinfo=UTC)  # Lundi, 6 h à Montréal.
    wednesday = datetime(2026, 11, 4, 15, 0, tzinfo=UTC)

    assert not weekly_due(db, niche, early)
    assert weekly_due(db, niche, MONDAY)
    assert weekly_due(db, niche, wednesday)  # L'ordinateur était éteint lundi.

    db.add(WeeklyReport(niche_id=niche.id, week_start="2026-11-02"))
    db.commit()
    assert not weekly_due(db, niche, wednesday)
    assert weekly_due(db, niche, wednesday + timedelta(days=7))


def weekly_claude(pattern_ids: list[int], count: int = 5, **kwargs) -> FakeWeeklyClaude:
    drafts = [script_draft(pattern_ids, angle=f"Angle {index}") for index in range(count)]
    return FakeWeeklyClaude(drafts=drafts, **kwargs)


def test_weekly_plan_has_five_scripts_with_one_exploration(db: Session, niche: Niche) -> None:
    pattern = add_pattern(db)
    fake = weekly_claude([pattern.id])

    [report_id] = run_weekly(services(fake), now=MONDAY)

    db.expire_all()
    report = db.get(WeeklyReport, report_id)
    assert len(report.script_ids) == 5
    scripts = [db.get(Script, item) for item in report.script_ids]
    assert [load_content(item).experiment for item in scripts].count(True) == 1
    assert report.week_start == "2026-11-02"
    assert "(exploration)" in report.markdown
    assert report.plan_error is None
    assert len(notifications(db, "weekly_report")) == 1
    path = get_settings().data_dir.resolve() / "reports" / f"week_2026-11-02_niche_{niche.id}.md"
    assert path.is_file()


def test_weekly_without_stats_does_not_pay_for_a_report(db: Session, niche: Niche) -> None:
    pattern = add_pattern(db)
    fake = weekly_claude([pattern.id])

    [report_id] = run_weekly(services(fake), now=MONDAY)

    assert fake.count("weekly_report") == 0
    db.expire_all()
    report = db.get(WeeklyReport, report_id)
    assert report.learnings is None
    assert "Aucune stat" in report.markdown


def test_weekly_report_keeps_only_claims_backed_by_a_video_of_the_week(
    db: Session, niche: Niche
) -> None:
    pattern = add_pattern(db)
    history(db, niche)
    script = add_script(db, niche, [pattern.id])
    row = save_performance(db, script, stats(500, days_ago=1))
    fake = weekly_claude(
        [pattern.id],
        learned=[
            {"statement": "Le hook a gagné", "video_ids": [row.video_id]},
            {"statement": "Claim sans preuve", "video_ids": [9999]},
        ],
    )

    [report_id] = run_weekly(services(fake), now=datetime.now(UTC))

    assert fake.count("weekly_report") == 1
    db.expire_all()
    report = db.get(WeeklyReport, report_id)
    assert [item["statement"] for item in report.learnings["learned"]] == ["Le hook a gagné"]
    assert f"[vidéo {row.video_id}]" in report.markdown
    assert "Claim sans preuve" not in report.markdown


def test_weekly_catches_up_missing_post_mortems(db: Session, niche: Niche) -> None:
    pattern = add_pattern(db)
    history(db, niche)
    script = add_script(db, niche, [pattern.id])
    save_performance(db, script, stats(500, days_ago=1))
    fake = weekly_claude([pattern.id])

    run_weekly(services(fake), now=datetime.now(UTC))

    assert fake.count("post_mortem") == 1
    assert db.scalar(select(func.count()).select_from(PostMortem)) == 1


def test_weekly_never_pays_twice_for_the_same_week(db: Session, niche: Niche) -> None:
    pattern = add_pattern(db)
    fake = weekly_claude([pattern.id])

    first = run_weekly(services(fake), now=MONDAY)
    second = run_weekly(services(fake), now=MONDAY + timedelta(hours=3))

    assert len(first) == 1
    assert second == []
    assert fake.count("script_ideas") == 1


def test_weekly_crash_after_starting_is_not_replayed(
    db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Le rapport existe avant le plan : un crash ne relance pas 5 $ de scripts au redémarrage."""
    pattern = add_pattern(db)
    monkeypatch.setattr(routines, "_content_plan", lambda *_: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        run_weekly(services(weekly_claude([pattern.id])), now=MONDAY)

    assert not weekly_due(db, niche, MONDAY + timedelta(hours=1))


def test_weekly_without_budget_says_so_and_writes_no_script(
    db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch
) -> None:
    pattern = add_pattern(db)
    small_budget(monkeypatch, 1000)
    fake = weekly_claude([pattern.id])

    [report_id] = run_weekly(services(fake), now=MONDAY)

    assert fake.calls == []
    db.expire_all()
    report = db.get(WeeklyReport, report_id)
    assert report.script_ids == []
    assert "Budget" in report.plan_error
    assert "Plan non généré" in report.markdown
    assert len(notifications(db, "budget")) == 1


def test_weekly_plan_can_be_turned_off(
    db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_WEEKLY_SCRIPTS", "0")
    get_settings.cache_clear()
    fake = weekly_claude([])

    [report_id] = run_weekly(services(fake), now=MONDAY)

    assert fake.calls == []
    assert "Aucun script prévu" in db.get(WeeklyReport, report_id).markdown


def test_weekly_reports_a_generation_failure(db: Session, niche: Niche) -> None:
    fake = weekly_claude([])  # Bibliothèque de patterns vide.

    [report_id] = run_weekly(services(fake), now=MONDAY)

    db.expire_all()
    assert "bibliothèque" in db.get(WeeklyReport, report_id).plan_error


def drop(path: Path, source: Path, name: str = "clip.mp4", mine: bool = False) -> Path:
    root = inbox_root()
    target = (root / "mine" if mine else root) / name
    shutil.copy(source, target)
    return target


def test_inbox_waits_until_the_file_stops_growing(
    db: Session, niche: Niche, sample_video: Path
) -> None:
    fake = FakeAnalysisClaude()
    path = drop(Path(sample_video), sample_video)
    inbox = Inbox()

    assert inbox.scan(services(fake)) == []  # Premier passage : on note la taille.
    with path.open("ab") as handle:
        handle.write(b"\0")  # Une copie est encore en cours.
    assert inbox.scan(services(fake)) == []
    assert db.scalar(select(func.count()).select_from(Video)) == 0

    Path(path).write_bytes(Path(sample_video).read_bytes())
    inbox.scan(services(fake))
    assert len(inbox.scan(services(fake))) == 1


def test_inbox_ingests_analyzes_and_files_away(
    db: Session, niche: Niche, sample_video: Path
) -> None:
    path = drop(sample_video, sample_video)

    [video_id] = Inbox().scan(services(), wait_for_copy=False)

    db.expire_all()
    video = db.get(Video, video_id)
    assert video.is_own_content is False
    assert video.niche_source == niche.name
    assert db.scalar(select(func.count()).select_from(Analysis)) == 1
    assert not path.exists()
    assert (inbox_root() / "done" / path.name).exists()
    assert len(notifications(db, "inbox_done")) == 1


def test_mine_folder_marks_the_video_as_mine(db: Session, niche: Niche, sample_video: Path) -> None:
    drop(sample_video, sample_video, mine=True)

    [video_id] = Inbox().scan(services(), wait_for_copy=False)

    db.expire_all()
    assert db.get(Video, video_id).is_own_content is True
    [notification] = notifications(db, "inbox_done")
    assert f"--video {video_id}" in notification.body


def test_inbox_ignores_other_files(db: Session, niche: Niche) -> None:
    (inbox_root() / "notes.txt").write_text("hello")

    assert Inbox().scan(services(), wait_for_copy=False) == []
    assert (inbox_root() / "notes.txt").exists()


def test_same_video_dropped_twice_is_not_paid_twice(
    db: Session, niche: Niche, sample_video: Path
) -> None:
    fake = FakeAnalysisClaude()
    drop(sample_video, sample_video)
    Inbox().scan(services(fake), wait_for_copy=False)
    calls = len(fake.calls)
    drop(sample_video, sample_video, "copie.mp4")

    Inbox().scan(services(fake), wait_for_copy=False)

    assert len(fake.calls) == calls
    assert (inbox_root() / "done" / "copie.mp4").exists()


def test_broken_file_goes_to_failed_with_a_notification(db: Session, niche: Niche) -> None:
    path = inbox_root() / "broken.mp4"
    path.write_bytes(b"pas une video")

    assert Inbox().scan(services(), wait_for_copy=False) == []

    assert (inbox_root() / "failed" / "broken.mp4").exists()
    assert len(notifications(db, "inbox_failed")) == 1


def test_inbox_keeps_files_when_the_budget_is_out(
    db: Session, niche: Niche, sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = drop(sample_video, sample_video)
    small_budget(monkeypatch, 1000)
    spend(db, 900)

    assert Inbox().scan(services(), wait_for_copy=False) == []

    assert path.exists()
    assert len(notifications(db, "budget")) == 1


def test_failed_analysis_stays_pending_for_the_daily_routine(
    db: Session, niche: Niche, sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = drop(sample_video, sample_video)

    def fail(*_, **__):
        raise LookupError("Analyse cassée")

    monkeypatch.setattr(routines, "analyze", fail)

    assert Inbox().scan(services(), wait_for_copy=False) == []

    assert not path.exists()
    assert (inbox_root() / "done" / path.name).exists()
    assert len(notifications(db, "inbox_failed")) == 1


def test_inbox_skips_a_pass_while_another_routine_works(
    db: Session, niche: Niche, sample_video: Path
) -> None:
    path = drop(sample_video, sample_video)

    with paid_work:
        assert Inbox().scan(services(), wait_for_copy=False) == []

    assert path.exists()


def test_empty_inbox_writes_nothing_to_the_journal(db: Session) -> None:
    Inbox().scan(services(), wait_for_copy=False)

    assert db.scalar(select(func.count()).select_from(AgentAction)) == 0


def test_scheduler_has_every_routine() -> None:
    scheduler = build_scheduler()

    assert {job.id for job in scheduler.get_jobs()} == {
        "daily",
        "weekly",
        "reminders",
        "inbox",
        "catch_up",
    }


def test_weekly_job_runs_on_monday_morning() -> None:
    scheduler = build_scheduler()

    trigger = str(scheduler.get_job("weekly").trigger)

    assert "day_of_week='mon'" in trigger
    assert "hour='7'" in trigger


def test_api_starts_and_stops_the_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_SCHEDULER_ENABLED", "true")
    monkeypatch.setenv("INGEST_WORKER_ENABLED", "false")
    get_settings.cache_clear()
    started = threading.Event()
    real = build_scheduler

    def spy():
        scheduler = real()
        scheduler.remove_all_jobs()
        started.set()
        return scheduler

    monkeypatch.setattr("app.main.build_scheduler", spy)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert started.is_set()


runner = CliRunner()


def test_cli_routines(db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "default_claude", FakeWeeklyClaude)
    monkeypatch.setattr(cli_module, "default_deps", make_deps)

    assert "rappel" in runner.invoke(cli, ["routine", "reminders"]).output
    assert "Vidéos analysées" in runner.invoke(cli, ["routine", "inbox"]).output
    assert runner.invoke(cli, ["routine", "daily", "--force"]).exit_code == 0
    assert runner.invoke(cli, ["routine", "nimporte"]).exit_code == 2
