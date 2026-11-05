"""Analyse de bout en bout : vraie ingestion ffmpeg, Claude simulé."""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app import cli as cli_module
from app.analysis.report import clock, render_report
from app.analysis.schemas import AnalysisDocument
from app.analysis.service import (
    AnalysisError,
    analyze,
    compute_metrics,
    pending_video_ids,
    report_path,
)
from app.cli import cli
from app.db.models import Analysis, Video
from app.ingest.jobs import ingest, submit
from app.llm.client import get_claude
from app.llm.errors import ClaudeCredentialsError
from app.main import app
from tests.analysis_helpers import FakeAnalysisClaude, claim, quote, sample_analysis
from tests.ingest_helpers import make_deps, make_video


@pytest.fixture
def video_id(db: Session, sample_video: Path) -> int:
    return ingest(str(sample_video), deps=make_deps()).video.id


def claims_in(node: Any) -> list[dict]:
    """Toutes les affirmations d'un JSON stocké."""
    if isinstance(node, dict):
        found = [node] if "statement" in node and "evidence" in node else []
        return found + [claim for value in node.values() for claim in claims_in(value)]
    if isinstance(node, list):
        return [claim for item in node for claim in claims_in(item)]
    return []


def test_analysis_is_stored_with_prompt_version(db: Session, video_id: int) -> None:
    fake = FakeAnalysisClaude()

    row = analyze(db, video_id, fake)

    assert row.prompt_version == "analysis/v1"
    assert row.video_id == video_id
    document = AnalysisDocument.model_validate(row.json_result)
    assert document.analysis.hook.score == 7
    assert document.verification.prompt_version == "analysis_verify/v1"
    assert [call["prompt"] for call in fake.calls] == ["analysis", "analysis_verify"]


def test_every_stored_claim_has_a_timestamped_proof(db: Session, video_id: int) -> None:
    draft = sample_analysis()
    draft["hook"]["visual"].append(claim("Inventé", quote("mots jamais prononcés")))

    row = analyze(db, video_id, FakeAnalysisClaude(draft))

    claims = claims_in(row.json_result["analysis"])
    assert len(claims) > 20
    assert "Inventé" not in [item["statement"] for item in claims]
    for item in claims:
        assert item["evidence"]
        for evidence in item["evidence"]:
            assert isinstance(evidence["start_s"], (int, float))
            assert isinstance(evidence["end_s"], (int, float))
            assert evidence["excerpt"]


def test_first_pass_gets_timeline_metadata_and_hook_frames_only(db: Session, video_id: int) -> None:
    fake = FakeAnalysisClaude()

    analyze(db, video_id, fake)

    first = fake.calls[0]
    assert json.loads(first["variables"]["timeline"])["version"] == 1
    assert "creator_followers" in json.loads(first["variables"]["metadata"])
    timestamps = first["variables"]["timestamps"]
    assert timestamps and max(timestamps) <= 5
    assert len(first["images"]) == len(timestamps)
    assert all(path.is_file() for path in first["images"])


def test_second_pass_rereads_the_draft_and_receives_code_findings(
    db: Session, video_id: int
) -> None:
    draft = sample_analysis()
    draft["hook"]["visual"].append(claim("Inventé", quote("mots jamais prononcés")))
    fake = FakeAnalysisClaude(draft)

    analyze(db, video_id, fake)

    second = fake.calls[1]
    assert "Inventé" in second["variables"]["analysis"]
    assert "Inventé" in second["variables"]["code_findings"]
    assert "images" not in second


def test_llm_verification_output_is_what_gets_stored(db: Session, video_id: int) -> None:
    verified = sample_analysis()
    verified["hook"]["visual"] = []
    fake = FakeAnalysisClaude(
        verified=verified,
        removals=[{"path": "hook.visual[0]", "statement": "Fond uni", "reason": "non prouvé"}],
        corrections=[{"path": "hook.score", "change": "7 -> 6"}],
    )

    document = AnalysisDocument.model_validate(analyze(db, video_id, fake).json_result)

    assert document.analysis.hook.visual == []
    assert document.verification.llm_removals[0].reason == "non prouvé"
    assert document.verification.llm_corrections[0].change == "7 -> 6"


def test_code_has_the_last_word_when_the_second_pass_misses_a_fake(
    db: Session, video_id: int
) -> None:
    verified = sample_analysis()
    verified["engagement"]["saves"] = [claim("Laissé par la passe 2", quote("citation fausse"))]

    row = analyze(db, video_id, FakeAnalysisClaude(verified=verified))

    document = AnalysisDocument.model_validate(row.json_result)
    assert document.analysis.engagement.saves == []
    assert document.verification.code_removals[0].statement == "Laissé par la passe 2"


def test_wrong_timestamp_is_corrected_in_the_stored_analysis(db: Session, video_id: int) -> None:
    verified = sample_analysis()
    verified["hook"]["spoken_text"] = [claim("Décalé", quote(start=7.0, end=7.5))]

    row = analyze(db, video_id, FakeAnalysisClaude(verified=verified))

    document = AnalysisDocument.model_validate(row.json_result)
    assert document.analysis.hook.spoken_text[0].evidence[0].start_s == pytest.approx(2.6)
    assert document.verification.code_corrections


def test_second_call_reuses_the_analysis_without_paying(db: Session, video_id: int) -> None:
    fake = FakeAnalysisClaude()
    first = analyze(db, video_id, fake)

    second = analyze(db, video_id, fake)

    assert second.id == first.id
    assert len(fake.calls) == 2


def test_force_redoes_the_analysis(db: Session, video_id: int) -> None:
    fake = FakeAnalysisClaude()
    first = analyze(db, video_id, fake)

    second = analyze(db, video_id, fake, force=True)

    assert second.id != first.id
    assert len(fake.calls) == 4


def test_second_pass_failure_stores_nothing_and_keeps_the_paid_draft(
    db: Session, video_id: int
) -> None:
    failing = FakeAnalysisClaude(fail_on="analysis_verify")

    with pytest.raises(RuntimeError):
        analyze(db, video_id, failing)

    assert db.scalars(select(Analysis)).all() == []

    recovered = FakeAnalysisClaude()
    analyze(db, video_id, recovered)
    assert recovered.count("analysis") == 0
    assert recovered.count("analysis_verify") == 1


def test_draft_file_is_removed_after_success(db: Session, video_id: int) -> None:
    row = analyze(db, video_id, FakeAnalysisClaude())

    folder = report_path(db.get(Video, video_id), row.id).parent
    assert list(folder.glob("analysis_draft_*")) == []


def test_unknown_video_raises(db: Session) -> None:
    with pytest.raises(LookupError):
        analyze(db, 999, FakeAnalysisClaude())


def test_video_not_ingested_is_refused_before_any_call(db: Session, sample_video: Path) -> None:
    video_id = submit(db, str(sample_video)).video_id
    fake = FakeAnalysisClaude()

    with pytest.raises(AnalysisError, match="non terminée"):
        analyze(db, video_id, fake)

    assert fake.calls == []


def test_duplicate_video_is_analyzed_through_its_reference(db: Session, video_id: int) -> None:
    twin = Video(duplicate_of_id=video_id)
    db.add(twin)
    db.commit()

    assert analyze(db, twin.id, FakeAnalysisClaude()).video_id == video_id


def test_metrics_are_computed_from_real_stats_only() -> None:
    video = Video(views=1000, shares=10, saves=30, comments=5, likes=None, creator_followers=500)

    metrics = compute_metrics(video)

    assert metrics.views_per_follower == 2
    assert metrics.share_rate == 0.01
    assert metrics.save_rate == 0.03
    assert metrics.comment_rate == 0.005
    assert metrics.like_rate is None


def test_metrics_are_none_without_stats_or_with_zero_views() -> None:
    assert compute_metrics(Video()).share_rate is None
    assert compute_metrics(Video(views=0, shares=0)).share_rate is None


def test_pending_lists_only_ingested_videos_not_yet_analyzed(
    db: Session, video_id: int, sample_video: Path, tmp_path: Path
) -> None:
    second = ingest(str(make_video(tmp_path / "b.mp4", "white")), deps=make_deps()).video.id
    submit(db, str(make_video(tmp_path / "c.mp4", "black")))  # Pas encore ingérée.

    assert pending_video_ids(db) == [video_id, second]

    analyze(db, video_id, FakeAnalysisClaude())

    assert pending_video_ids(db) == [second]


def test_report_file_is_written_and_readable(db: Session, video_id: int) -> None:
    row = analyze(db, video_id, FakeAnalysisClaude())

    report = report_path(db.get(Video, video_id), row.id).read_text("utf-8")

    assert report.startswith("# Pourquoi cette vidéo a marché")
    for title in ("Verdict", "Hook : 7/10", "Angle", "Structure", "Rétention", "Vérification"):
        assert f"## {title}" in report
    assert "[00:02.6–" in report
    assert "] double your views" in report
    assert "Résultat d'abord" in report


def test_report_shows_removed_claims_and_missing_stats(db: Session, video_id: int) -> None:
    verified = sample_analysis()
    verified["engagement"]["saves"] = [claim("Laissé", quote("citation fausse"))]

    row = analyze(db, video_id, FakeAnalysisClaude(verified=verified))
    report = render_report(AnalysisDocument.model_validate(row.json_result))

    assert "1 affirmation(s) retirée(s)" in report
    assert "Retiré : Laissé" in report
    assert "Partages / vues : non disponible" in report
    assert "Rien d'étayé sur ce point" in report


def test_report_survives_an_analysis_stripped_of_everything(db: Session, video_id: int) -> None:
    verified = sample_analysis()
    for section, name in (("angle", "alternatives"), ("structure", "beats")):
        verified[section][name] = []
    verified["verdict"]["reasons"] = []
    verified["emotion_curve"] = []

    row = analyze(db, video_id, FakeAnalysisClaude(verified=verified))

    assert "Aucune raison" in render_report(AnalysisDocument.model_validate(row.json_result))


def test_clock() -> None:
    assert clock(2.6) == "00:02.6"
    assert clock(75.3) == "01:15.3"
    assert clock(59.96) == "01:00.0"


client = TestClient(app)


@pytest.fixture
def api_claude() -> FakeAnalysisClaude:
    fake = FakeAnalysisClaude()
    app.dependency_overrides[get_claude] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


def test_post_then_get_analysis(db: Session, video_id: int, api_claude: FakeAnalysisClaude) -> None:
    created = client.post(f"/videos/{video_id}/analyze")

    assert created.status_code == 201
    body = client.get(f"/analyses/{created.json()['id']}").json()
    assert body["prompt_version"] == "analysis/v1"
    assert body["result"]["analysis"]["hook"]["score"] == 7
    assert body["report_markdown"].startswith("# Pourquoi cette vidéo a marché")


def test_api_errors(db: Session, sample_video: Path, api_claude: FakeAnalysisClaude) -> None:
    pending = submit(db, str(sample_video)).video_id

    assert client.post("/videos/999/analyze").status_code == 404
    assert client.post(f"/videos/{pending}/analyze").status_code == 409
    assert client.get("/analyses/999").status_code == 404


def test_api_reports_missing_key_as_503(db: Session, video_id: int) -> None:
    class NoKey(FakeAnalysisClaude):
        def call_json(self, **_: Any):
            raise ClaudeCredentialsError("Aucune clé API")

    app.dependency_overrides[get_claude] = lambda: NoKey()
    try:
        response = client.post(f"/videos/{video_id}/analyze")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "clé API" in response.json()["detail"]


runner = CliRunner()


@pytest.fixture
def cli_claude(monkeypatch: pytest.MonkeyPatch) -> FakeAnalysisClaude:
    fake = FakeAnalysisClaude()
    monkeypatch.setattr(cli_module, "default_claude", lambda: fake)
    return fake


def test_cli_analyze_one(db: Session, video_id: int, cli_claude: FakeAnalysisClaude) -> None:
    result = runner.invoke(cli, ["analyze", str(video_id)])

    assert result.exit_code == 0, result.output
    assert "analysis_1.md" in result.output.replace("\n", "")


def test_cli_analyze_all_pending(
    db: Session, video_id: int, tmp_path: Path, cli_claude: FakeAnalysisClaude
) -> None:
    ingest(str(make_video(tmp_path / "b.mp4", "white")), deps=make_deps())

    result = runner.invoke(cli, ["analyze", "--all-pending"])

    assert result.exit_code == 0, result.output
    assert "2/2" in result.output
    assert "Aucune vidéo" in runner.invoke(cli, ["analyze", "--all-pending"]).output


def test_cli_analyze_needs_exactly_one_target(db: Session, cli_claude: FakeAnalysisClaude) -> None:
    assert runner.invoke(cli, ["analyze"]).exit_code == 2
    assert runner.invoke(cli, ["analyze", "1", "--all-pending"]).exit_code == 2


def test_cli_reports_failure(db: Session, cli_claude: FakeAnalysisClaude) -> None:
    result = runner.invoke(cli, ["analyze", "999"])

    assert result.exit_code == 1
    assert "inconnue" in result.output
