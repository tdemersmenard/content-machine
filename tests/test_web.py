"""Dashboard : pages, éditions, tâches de fond et sécurité."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app.agent.chat import chat, start_session
from app.cli import cli
from app.config import get_settings
from app.db.models import (
    Analysis,
    IngestJob,
    Lesson,
    LessonScope,
    Niche,
    Notification,
    Performance,
    PostMortem,
    Script,
    ScriptStatus,
    Video,
    VideoPattern,
    WeeklyReport,
)
from app.generate.service import load_content, update_script
from app.ingest.jobs import ingest
from app.learning.service import save_performance
from app.main import app
from app.web import media as media_module
from app.web import routes
from app.web import views as views_module
from app.web.tasks import TaskRunner, runner
from tests.agent_helpers import FakeAgentClaude, FakeWeeklyClaude, text_turn, tool_turn
from tests.analysis_helpers import sample_analysis
from tests.ingest_helpers import make_deps
from tests.script_helpers import script_draft
from tests.test_learning import add_pattern, add_script, history, stats
from tests.test_niche import make_niche

client = TestClient(app, follow_redirects=False)


@pytest.fixture
def niche(db: Session) -> Niche:
    return make_niche(db)


@pytest.fixture(autouse=True)
def inline_tasks() -> Iterator[None]:
    runner.inline = True
    runner._tasks.clear()
    yield
    runner.inline = False


def analyzed_video(db: Session, sample_video: Path) -> Video:
    video_id = ingest(str(sample_video), deps=make_deps()).video.id
    pattern = add_pattern(db, "Promesse chiffrée")
    db.add(VideoPattern(video_id=video_id, pattern_id=pattern.id, role="opening"))
    document = {
        "analysis": sample_analysis(),
        "metrics": {
            "views_per_follower": None,
            "share_rate": None,
            "save_rate": None,
            "comment_rate": None,
            "like_rate": None,
        },
        "verification": {
            "prompt_version": "v",
            "model": "m",
            "llm_corrections": [],
            "llm_removals": [],
            "code_corrections": [],
            "code_removals": [],
        },
    }
    db.add(
        Analysis(video_id=video_id, prompt_version="analysis/v1", model="m", json_result=document)
    )
    db.commit()
    return db.get(Video, video_id)


@pytest.mark.parametrize(
    "path",
    [
        "/ui/inbox",
        "/ui/inbox/queue",
        "/ui/patterns",
        "/ui/studio",
        "/ui/results",
        "/ui/notifications",
        "/ui/login",
    ],
)
def test_pages_work_on_an_empty_database(db: Session, path: str) -> None:
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("path", ["/ui/brain", "/ui/chat"])
def test_pages_with_a_niche_work_without_data(db: Session, niche: Niche, path: str) -> None:
    assert client.get(path).status_code == 200


def test_root_goes_to_the_inbox(db: Session) -> None:
    assert client.get("/").headers["location"] == "/ui/inbox"
    assert client.get("/ui").headers["location"] == "/ui/inbox"


def test_every_page_is_mobile_ready(db: Session) -> None:
    html = client.get("/ui/inbox").text

    assert 'name="viewport"' in html
    assert "/static/app.css" in html
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/htmx.min.js").status_code == 200


def test_links_are_queued_and_shown(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes, "_analyze_when_ingested", lambda *_: None)

    response = client.post(
        "/ui/inbox/links", data={"links": "https://tiktok.com/a\npas-un-lien\nhttps://x.com/b"}
    )

    assert response.status_code == 200
    assert "Lien ignoré : pas-un-lien" in response.text
    assert db.scalar(select(func.count()).select_from(IngestJob)) == 2


def test_upload_queues_the_video_and_refuses_other_files(
    db: Session, sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(routes, "_analyze_when_ingested", lambda *_: None)

    response = client.post(
        "/ui/inbox/files",
        data={"own": "true"},
        files=[
            ("files", ("clip.mp4", sample_video.read_bytes(), "video/mp4")),
            ("files", ("virus.exe", b"x", "application/octet-stream")),
        ],
    )

    assert "Fichier ignoré : virus.exe" in response.text
    video = db.scalar(select(Video))
    assert video.is_own_content is True


def test_failed_analysis_shows_in_the_queue(
    db: Session, sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_id = ingest(str(sample_video), deps=make_deps()).video.id

    def boom(*_):
        raise RuntimeError("Budget épuisé")

    monkeypatch.setattr(routes, "_analyze_when_ingested", boom)

    html = client.post(f"/ui/videos/{video_id}/analyze").text

    assert "Analyse non faite" in html
    assert "Budget épuisé" in html


def test_queue_shows_analyzed_videos_with_a_link_to_adapt(db: Session, sample_video: Path) -> None:
    video = analyzed_video(db, sample_video)

    html = client.get("/ui/inbox/queue").text

    assert "Analysée" in html
    assert f"/ui/studio?video={video.id}" in html


def test_video_page_has_seekable_moments_and_pattern_badges(
    db: Session, sample_video: Path
) -> None:
    video = analyzed_video(db, sample_video)

    html = client.get(f"/ui/videos/{video.id}").text

    assert f'src="/ui/media/{video.id}"' in html
    assert "data-seek" in html
    assert "Pourquoi ça a marché" in html
    assert "Promesse chiffrée" in html
    assert "/ui/patterns#pattern-" in html
    assert "ne décrit" not in html
    assert "son potentiel" in html


def test_video_without_analysis_says_so(db: Session, sample_video: Path) -> None:
    video_id = ingest(str(sample_video), deps=make_deps()).video.id

    assert "Pas encore d'analyse" in client.get(f"/ui/videos/{video_id}").text


def test_stats_stub_videos_stay_out_of_the_queue(db: Session, niche: Niche) -> None:
    script = add_script(db, niche, [])
    save_performance(db, script, stats())

    assert "Rien dans la file" in client.get("/ui/inbox/queue").text


def test_unknown_video(db: Session) -> None:
    assert client.get("/ui/videos/999").status_code == 404
    assert client.get("/ui/media/999").status_code == 404


def test_moments_are_sorted_and_placed_on_the_timeline() -> None:
    moments = views_module.moments(sample_analysis(), 9.0)

    assert "hook" in {item["kind"] for item in moments}
    assert [item["start"] for item in moments] == sorted(item["start"] for item in moments)
    assert all(0 <= item["percent"] <= 100 for item in moments)


def test_browser_playable_video_is_served_with_range_support(
    db: Session, sample_video: Path
) -> None:
    video_id = ingest(str(sample_video), deps=make_deps()).video.id

    response = client.get(f"/ui/media/{video_id}", headers={"Range": "bytes=0-99"})

    assert response.status_code == 206
    assert len(response.content) == 100


def test_hevc_gets_a_preview_once(
    db: Session, sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = db.get(Video, ingest(str(sample_video), deps=make_deps()).video.id)
    monkeypatch.setattr(media_module, "_codec", lambda _: "hevc")

    first = media_module.playable_path(video)
    stamp = first.stat().st_mtime_ns
    second = media_module.playable_path(video)

    assert first.name == "preview.mp4"
    assert second.stat().st_mtime_ns == stamp


def test_patterns_filter_sort_and_show_confidence(db: Session, niche: Niche) -> None:
    add_pattern(db, "Jamais essayé", "hook")
    good = add_pattern(db, "Bien noté", "hook")
    good.score_alpha, good.score_beta = 4.0, 1.0
    add_pattern(db, "Autre type", "ending")
    db.commit()

    html = client.get("/ui/patterns?type=hook").text
    partial = client.get("/ui/patterns?type=ending", headers={"HX-Request": "true"}).text

    assert html.index("Bien noté") < html.index("Jamais essayé")
    assert "Autre type" not in html
    assert "Jamais testé" in html
    assert "Score 0.80" in html
    assert "<html" not in partial
    assert "Autre type" in partial


def test_studio_shows_scripts_side_by_side(db: Session, niche: Niche) -> None:
    first, second = add_script(db, niche, []), add_script(db, niche, [])

    html = client.get(f"/ui/studio?ids={first.id},{second.id}").text

    assert f'id="script-{first.id}"' in html
    assert f'id="script-{second.id}"' in html
    assert "Approuver" in html


def test_studio_runs_adapt_in_the_background_and_redirects(
    db: Session, niche: Niche, sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = analyzed_video(db, sample_video)
    pattern_id = db.scalar(select(VideoPattern.pattern_id))
    fake = FakeAgentClaude(
        drafts=[script_draft([pattern_id], angle=f"Angle {index}") for index in range(3)]
    )
    monkeypatch.setattr(routes, "claude_client", lambda: fake)

    response = client.post(
        "/ui/studio/adapt", data={"video_id": video.id, "niche_id": niche.id, "variants": 3}
    )
    task_id = (
        response.text.split("/ui/tasks/")[1].split('"')[0]
        if "/ui/tasks/" in response.text
        else None
    )
    done = next(item for item in runner._tasks.values() if item.label.startswith("3 variante"))

    assert response.status_code == 200
    assert task_id is None  # Déjà finie : plus besoin d'interroger.
    assert done.status == "done"
    finished = client.get(f"/ui/tasks/{done.id}")
    assert finished.headers["HX-Redirect"].startswith("/ui/studio?ids=")
    assert db.scalar(select(func.count()).select_from(Script)) == 3


def test_task_failure_is_shown(db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_DAILY_TOKEN_BUDGET", "10")
    get_settings.cache_clear()

    html = client.post("/ui/studio/adapt", data={"video_id": 1, "niche_id": niche.id}).text

    assert "Échec" in html
    assert "Budget" in html


def test_lost_task_says_so(db: Session) -> None:
    assert "redémarré" in client.get("/ui/tasks/inconnue").text


def test_editing_a_beat_updates_duration_and_checks(db: Session, niche: Niche) -> None:
    script = add_script(db, niche, [])
    before = load_content(script).estimated_duration_s

    response = client.post(
        f"/ui/scripts/{script.id}/beats/0",
        data={"spoken": "Un texte bien plus long [À VÉRIFIER : le montant] " + "mot " * 30},
    )

    assert response.status_code == 200
    assert "le montant" in response.text
    db.expire_all()
    content = load_content(db.get(Script, script.id))
    assert content.estimated_duration_s > before
    assert content.to_verify == ["le montant"]


def test_clearing_the_marker_clears_the_warning(db: Session, niche: Niche) -> None:
    script = add_script(db, niche, [])
    url = f"/ui/scripts/{script.id}/beats/0"
    client.post(url, data={"spoken": "Texte [À VÉRIFIER : x]"})

    response = client.post(url, data={"spoken": "Texte corrigé"})

    assert "À vérifier" not in response.text


def test_empty_beat_and_unknown_beat_are_refused(db: Session, niche: Niche) -> None:
    script = add_script(db, niche, [])

    assert client.post(f"/ui/scripts/{script.id}/beats/0", data={"spoken": " "}).status_code == 422
    assert client.post(f"/ui/scripts/{script.id}/beats/99", data={"spoken": "x"}).status_code == 404
    assert client.post("/ui/scripts/999/beats/0", data={"spoken": "x"}).status_code == 404


def test_choosing_a_hook_moves_it_first(db: Session, niche: Niche) -> None:
    script = add_script(db, niche, [])
    wanted = script.content["hooks"][2]["spoken"]

    response = client.post(f"/ui/scripts/{script.id}/hook/2")

    assert response.status_code == 200
    db.expire_all()
    assert db.get(Script, script.id).content["hooks"][0]["spoken"] == wanted


def test_approve_then_teleprompter_and_print(db: Session, niche: Niche) -> None:
    script = add_script(db, niche, [])

    html = client.post(f"/ui/scripts/{script.id}/approve").text

    assert "Approuvé" in html
    assert "Approuver" not in html
    db.expire_all()
    assert db.get(Script, script.id).status is ScriptStatus.APPROVED

    content = load_content(script)
    prompter = client.get(f"/ui/scripts/{script.id}/teleprompter").text
    assert content.beats[0].spoken in prompter
    assert content.hooks[0].spoken in prompter
    assert f"const duration = {content.estimated_duration_s}" in prompter
    printed = client.get(f"/ui/scripts/{script.id}/print").text
    assert "window.print()" in printed
    assert content.caption in printed
    assert client.get(f"/ui/scripts/{script.id}/markdown").status_code == 200


def test_verify_markers_stand_out_on_the_teleprompter(db: Session, niche: Niche) -> None:
    script = add_script(db, niche, [])
    client.post(f"/ui/scripts/{script.id}/beats/0", data={"spoken": "Il y a [À VÉRIFIER : x] ici"})

    assert (
        "<mark>[À VÉRIFIER : x]</mark>" in client.get(f"/ui/scripts/{script.id}/teleprompter").text
    )


def test_results_lists_ready_and_published_scripts(db: Session, niche: Niche) -> None:
    ready = add_script(db, niche, [])
    update_script(db, ready, status=ScriptStatus.APPROVED)
    history(db, niche)
    winner = add_script(db, niche, [])
    save_performance(db, winner, stats(500, days_ago=1))

    html = client.get("/ui/results").text

    assert "Prêts à publier" in html
    assert "C&#39;est publié" in html or "C'est publié" in html
    assert "Winner" in html


def test_publish_button_starts_the_reminders(db: Session, niche: Niche) -> None:
    script = add_script(db, niche, [])
    update_script(db, script, status=ScriptStatus.APPROVED)

    response = client.post(f"/ui/scripts/{script.id}/publish")

    assert response.headers["location"] == f"/ui/results/{script.id}"
    db.expire_all()
    assert db.get(Script, script.id).published_at is not None


def test_quick_stats_save_and_read_like_a_phone_would_type_them(
    db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(routes, "claude_client", FakeWeeklyClaude)
    history(db, niche)
    script = add_script(db, niche, [])

    response = client.post(
        f"/ui/results/{script.id}",
        data={"horizon_h": "72", "views": "12,5k", "retention_3s": "68", "shares": ""},
    )

    assert response.status_code == 303
    db.expire_all()
    row = db.scalar(select(Performance).where(Performance.script_id == script.id))
    assert row.views == 12500
    assert row.retention_3s == 0.68
    assert row.shares is None
    assert db.scalar(select(func.count()).select_from(PostMortem)) == 1
    html = client.get(f"/ui/results/{script.id}").text
    assert "Diagnostic" in html
    assert "À garder" in html


def test_unreadable_stats_do_not_save(db: Session, niche: Niche) -> None:
    script = add_script(db, niche, [])

    response = client.post(f"/ui/results/{script.id}", data={"views": "beaucoup"})

    assert "error=" in response.headers["location"]
    assert db.scalar(select(func.count()).select_from(Performance)) == 0


def test_stats_can_link_the_ingested_video(
    db: Session, niche: Niche, sample_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(routes, "claude_client", FakeWeeklyClaude)
    video_id = ingest(str(sample_video), is_own_content=True, deps=make_deps()).video.id
    script = add_script(db, niche, [])

    client.post(
        f"/ui/results/{script.id}",
        data={"horizon_h": "24", "views": "100", "published_video_id": str(video_id)},
    )

    db.expire_all()
    assert db.get(Script, script.id).published_video_id == video_id


def test_brain_shows_lessons_chart_prediction_and_report(db: Session, niche: Niche) -> None:
    history(db, niche)
    video_ids = []
    for index, value in enumerate((50, 80, 300, 400)):
        script = add_script(db, niche, [])
        video_ids.append(save_performance(db, script, stats(value, days_ago=10 - index)).video_id)
    db.add(
        Lesson(
            scope=LessonScope.NICHE,
            niche_id=niche.id,
            statement="Le hook court gagne",
            evidence=[{"video_id": video_ids[0]}, {"pattern_id": 1, "effect": "avoid"}],
            confidence=0.7,
        )
    )
    db.add(WeeklyReport(niche_id=niche.id, week_start="2026-11-02", markdown="# Bilan\n\nRien."))
    db.commit()

    html = client.get("/ui/brain").text

    assert "Le hook court gagne" in html
    assert f'href="/ui/videos/{video_ids[0]}"' in html
    assert "Écarte le pattern 1" in html
    assert "<polyline" in html
    assert "Voir le tableau" in html
    assert "% " in html
    assert "<h1>Bilan</h1>" in html


def test_trends_need_four_results_and_compare_recent_to_old(db: Session, niche: Niche) -> None:
    rising = add_pattern(db, "En hausse")
    falling = add_pattern(db, "En baisse")
    rare = add_pattern(db, "Rare")
    history(db, niche)
    for index, value in enumerate((30, 30, 500, 500)):
        script = add_script(db, niche, [rising.id])
        save_performance(db, script, stats(value, days_ago=20 - index))
    for index, value in enumerate((500, 500, 30, 30)):
        script = add_script(db, niche, [falling.id])
        save_performance(db, script, stats(value, days_ago=30 - index))
    script = add_script(db, niche, [rare.id])
    save_performance(db, script, stats(500, days_ago=1))

    trends = views_module.pattern_trends(db, niche.id)

    assert [item["name"] for item in trends["rising"]] == ["En hausse"]
    assert [item["name"] for item in trends["falling"]] == ["En baisse"]


def test_chat_runs_and_shows_answer_with_links(
    db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch
) -> None:
    pattern = add_pattern(db)
    fake = FakeAgentClaude(
        [tool_turn("get_niche", say="Je regarde."), text_turn(f"Voir [pattern {pattern.id}].")]
    )
    monkeypatch.setattr(routes, "claude_client", lambda: fake)

    client.post("/ui/chat", data={"message": "Salut"})
    task = next(
        item for item in reversed(runner._tasks.values()) if item.label.startswith("L'agent")
    )

    assert task.status == "done"
    assert "get_niche" in task.steps
    html = client.get(task.redirect).text
    assert "Salut" in html
    assert f'href="/ui/patterns#pattern-{pattern.id}"' in html
    assert "get_niche" in html
    assert "Mémoire au début" not in html


def test_chat_warns_about_invented_references(db: Session, niche: Niche) -> None:
    fake = FakeAgentClaude([text_turn("Selon [vidéo 777], oui.")])
    agent_session = start_session(db, niche.id)
    chat(db, agent_session, "x", fake)

    html = client.get(f"/ui/chat?session={agent_session.id}").text

    assert "Références introuvables en base : video 777" in html


def test_chat_continues_the_same_session(
    db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeAgentClaude([text_turn("Un"), text_turn("Deux")])
    monkeypatch.setattr(routes, "claude_client", lambda: fake)
    agent_session = start_session(db, niche.id)
    chat(db, agent_session, "x", fake)

    client.post("/ui/chat", data={"message": "y", "session_id": agent_session.id})

    assert len(fake.tool_calls[-1]["messages"]) == 3
    assert client.post("/ui/chat", data={"message": "y", "session_id": 999}).status_code == 404


def test_html_from_claude_or_a_video_cannot_run() -> None:
    rendered = str(views_module.render_markdown("<script>alert(1)</script> [vidéo 3]"))

    assert "<script>" not in rendered
    assert 'href="/ui/videos/3"' in rendered


def test_chat_page_escapes_the_user_message(db: Session, niche: Niche) -> None:
    fake = FakeAgentClaude([text_turn("Ok")])
    agent_session = start_session(db, niche.id)
    chat(db, agent_session, "<img src=x onerror=alert(1)>", fake)

    html = client.get(f"/ui/chat?session={agent_session.id}").text

    assert "<img src=x" not in html


def test_notifications_show_in_the_nav_and_can_be_read(db: Session, niche: Niche) -> None:
    script = add_script(db, niche, [])
    row = Notification(
        kind="stats_reminder", title="Stats à 24 h", script_id=script.id, horizon_h=24
    )
    db.add(row)
    db.commit()

    assert '<span class="badge badge-busy">1</span>' in client.get("/ui/inbox").text
    assert f"/ui/results/{script.id}" in client.get("/ui/notifications").text
    assert client.post(f"/ui/notifications/{row.id}/read").status_code == 303
    assert '<span class="badge badge-busy">' not in client.get("/ui/inbox").text
    assert client.post("/ui/notifications/999/read").status_code == 404


def test_writes_from_another_site_are_refused(db: Session) -> None:
    response = client.post(
        "/ui/inbox/links",
        data={"links": "https://x.com/a"},
        headers={"Origin": "https://mechant.example"},
    )

    assert response.status_code == 403
    assert db.scalar(select(func.count()).select_from(IngestJob)) == 0


def test_same_origin_writes_pass(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes, "_analyze_when_ingested", lambda *_: None)

    response = client.post(
        "/ui/inbox/links",
        data={"links": "https://x.com/a"},
        headers={"Origin": "http://testserver"},
    )

    assert response.status_code == 200


@pytest.fixture
def locked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHBOARD_PASSWORD", "secret")
    get_settings.cache_clear()


def test_password_protects_pages_and_the_api(db: Session, locked: None) -> None:
    assert client.get("/ui/inbox").headers["location"] == "/ui/login"
    assert client.get("/agent/budget").status_code == 401
    assert client.get("/health").status_code == 200
    assert client.get("/static/app.css").status_code == 200


def test_login_opens_the_dashboard(db: Session, locked: None) -> None:
    other = TestClient(app, follow_redirects=False)

    assert "incorrect" in other.post("/ui/login", data={"password": "non"}).text
    response = other.post("/ui/login", data={"password": "secret"})

    assert response.status_code == 303
    assert other.get("/ui/inbox").status_code == 200


def test_this_computer_does_not_need_the_password(db: Session, locked: None) -> None:
    local = TestClient(app, client=("127.0.0.1", 5000), follow_redirects=False)

    assert local.get("/ui/inbox").status_code == 200


def test_lan_mode_requires_a_password() -> None:
    result = CliRunner().invoke(cli, ["dev", "--lan"])

    assert result.exit_code == 1
    assert "DASHBOARD_PASSWORD" in result.output


def test_dev_runs_migrations_and_warns_before_the_weekly_plan(
    db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import cli as cli_module

    monkeypatch.setenv("AGENT_SCHEDULER_ENABLED", "true")
    get_settings.cache_clear()
    calls = []
    monkeypatch.setattr(cli_module.command, "upgrade", lambda *_: calls.append("upgrade"))
    monkeypatch.setattr(cli_module.uvicorn, "run", lambda *a, **k: calls.append(k["host"]))
    monkeypatch.setattr(cli_module, "_weekly_due", lambda _: [niche])

    result = CliRunner().invoke(cli, ["dev"])

    assert result.exit_code == 0, result.output
    assert calls == ["upgrade", "127.0.0.1"]
    assert "plan de la semaine" in result.output


def test_task_runner_keeps_a_bounded_history() -> None:
    local = TaskRunner()
    local.inline = True

    for index in range(250):
        local.start(str(index), lambda _: None)

    assert len(local._tasks) == 200
    assert local.running_key("x") is None
