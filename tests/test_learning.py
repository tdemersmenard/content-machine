"""Stats, scores normalisés, patterns, post-mortems, leçons et exploration."""

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app import cli as cli_module
from app.cli import cli
from app.config import get_settings
from app.db.models import (
    Lesson,
    LessonScope,
    Niche,
    Pattern,
    PatternScore,
    Performance,
    PostMortem,
    Script,
    ScriptStatus,
    SynthesisRun,
    Transcript,
    Video,
)
from app.generate.report import render_script
from app.generate.schemas import ScriptContent
from app.generate.service import (
    active_lessons,
    exploration_combination,
    exploration_slots,
    generate,
    load_content,
    thompson_combination,
    update_script,
)
from app.learning.imports import (
    ImportError_,
    import_csv,
    parse_count,
    parse_rate,
    parse_seconds,
)
from app.learning.lessons import maybe_synthesize
from app.learning.report import build_report, prediction_accuracy
from app.learning.schemas import PerformanceInput
from app.learning.scoring import (
    baselines,
    classify,
    composite_score,
    decay_weight,
    performance_indexes,
    weights_for,
)
from app.learning.service import (
    rebuild_pattern_scores,
    record_performance,
    save_performance,
    scored_results,
)
from app.llm.client import ClaudeResult, get_claude
from app.llm.prompts import load_prompt
from app.main import app
from app.niche.profile import load_profile
from app.patterns.search import niche_scores, score_mean, top_patterns
from tests.script_helpers import FakeScriptClaude, script_draft
from tests.test_niche import make_niche

NOW = datetime.now(UTC)


class FakeLearningClaude:
    def __init__(self, lessons: list[dict] | None = None, contradicted=None, fail_on=None) -> None:
        self.lessons = lessons or []
        self.contradicted = contradicted or []
        self.fail_on = fail_on
        self.calls: list[dict[str, Any]] = []

    def call_json(self, *, prompt: str, version: str, schema: Any, model: str, **kwargs: Any):
        self.calls.append({"prompt": prompt, **kwargs})
        if prompt == self.fail_on:
            raise RuntimeError("Panne simulée")
        if prompt == "post_mortem":
            data = schema(
                summary="Résumé du diagnostic",
                causes=[
                    {
                        "category": "hook",
                        "weight_percent": 100,
                        "explanation": "Le hook",
                        "evidence": "retention_3s",
                    }
                ],
                script_vs_actual="Non ingérée",
                keep=["Garder"],
                change=["Changer"],
                prediction_review="Trop optimiste",
            )
        else:
            data = schema(lessons=self.lessons, contradicted=self.contradicted)
        return ClaudeResult(
            data=data, model=model, prompt_version=f"{prompt}/{version}", attempts=1, cost_usd=0.01
        )

    def count(self, prompt: str) -> int:
        return sum(1 for call in self.calls if call["prompt"] == prompt)


@pytest.fixture
def niche(db: Session) -> Niche:
    return make_niche(db)


def add_pattern(db: Session, name: str = "P", kind: str = "hook") -> Pattern:
    pattern = Pattern(type=kind, name=name, description="d", template="[x]", examples=[])
    db.add(pattern)
    db.commit()
    return pattern


def add_script(
    db: Session, niche: Niche, pattern_ids: list[int], predicted: float | None = 60.0
) -> Script:
    content = ScriptContent(
        **script_draft(pattern_ids),
        mode="generate",
        topic_ids=[],
        estimated_duration_s=10,
        words_per_second=3,
        duration_source="default",
        to_verify=[],
        critiques=[],
        final_critique={"hook_score": 7, "swipe_points": [], "fixes": []},
        predicted_score={"value": 62, "hook_score": 7, "pattern_mean": 0.5},
        prompt_versions={},
    )
    script = Script(
        niche_id=niche.id,
        pattern_ids=pattern_ids,
        prompt_version="script_write/v1",
        content=content.model_dump(mode="json"),
        predicted_score=predicted,
    )
    db.add(script)
    db.commit()
    return script


def stats(value: float = 100, days_ago: float = 0, horizon: int = 72, **extra: Any):
    return PerformanceInput(
        horizon_h=horizon,
        measured_at=NOW - timedelta(days=days_ago),
        views=int(value * 100),
        shares=int(value),
        retention_3s=min(value / 1000, 1),
        **extra,
    )


def history(db: Session, niche: Niche, count: int = 3, value: float = 100, horizon: int = 72):
    """Des vidéos de référence, plus anciennes que tout le reste."""
    for index in range(count):
        script = add_script(db, niche, [])
        save_performance(db, script, stats(value, days_ago=100 - index, horizon=horizon))


def test_weights_favor_retention_and_shares_over_likes() -> None:
    weights = weights_for(None)

    assert weights["retention_3s"] > weights["likes"]
    assert weights["shares"] > weights["likes"]


def test_weights_follow_the_goal_and_can_be_overridden() -> None:
    assert weights_for("followers")["follows"] > weights_for(None)["follows"]
    assert weights_for("followers", {"follows": 9.0})["follows"] == 9.0


def test_baseline_is_the_median_of_the_last_20() -> None:
    rows = [{"views": 1_000_000}] * 5 + [{"views": value} for value in range(1, 21)]

    assert baselines(rows, 3)["views"] == 10.5


def test_baseline_needs_enough_values_for_a_metric() -> None:
    rows = [{"views": 1, "likes": None}, {"views": 2, "likes": 3}, {"views": 3, "likes": None}]

    assert baselines(rows, 3) == {"views": 2}


def test_indexes_skip_missing_values_and_zero_baselines() -> None:
    indexes = performance_indexes(
        {"views": 300, "likes": None, "shares": 5}, {"views": 100, "likes": 10, "shares": 0}
    )

    assert indexes == {"views": 3.0}


def test_composite_ignores_missing_metrics_instead_of_counting_them_as_zero() -> None:
    assert composite_score({"views": 2.0}, {"views": 1.0, "likes": 5.0}) == 2.0
    assert composite_score({"views": 2.0, "shares": 1.0}, {"views": 1.0, "shares": 3.0}) == 1.25
    assert composite_score({}, {"views": 1.0}) is None


@pytest.mark.parametrize(
    ("value", "label"),
    [(1.51, "winner"), (1.5, "neutral"), (1.0, "neutral"), (0.7, "neutral"), (0.69, "loser")],
)
def test_classification(value: float, label: str) -> None:
    assert classify(value) == label


def test_decay_halves_after_one_half_life() -> None:
    assert decay_weight(0, 90) == 1
    assert decay_weight(90, 90) == pytest.approx(0.5)
    assert decay_weight(-5, 90) == 1


def test_first_videos_are_kept_without_a_score(db: Session, niche) -> None:
    script = add_script(db, niche, [])

    row = save_performance(db, script, stats())

    assert row.views == 10000
    assert row.composite is None
    assert row.label is None


def test_score_compares_to_previous_videos_only(db: Session, niche) -> None:
    history(db, niche)
    script = add_script(db, niche, [])

    row = save_performance(db, script, stats(300))

    assert row.composite == pytest.approx(3.0)
    assert row.label == "winner"
    assert row.baseline_size == 3
    assert row.indexes["views"] == pytest.approx(3.0)


def test_only_the_same_horizon_is_compared(db: Session, niche) -> None:
    history(db, niche, value=1000, horizon=168)
    script = add_script(db, niche, [])

    assert save_performance(db, script, stats(100, horizon=24)).composite is None


def test_several_readings_for_one_video(db: Session, niche) -> None:
    script = add_script(db, niche, [])

    for horizon in (24, 72, 168):
        save_performance(db, script, stats(horizon=horizon))

    rows = db.scalars(select(Performance).where(Performance.script_id == script.id)).all()
    assert sorted(row.horizon_h for row in rows) == [24, 72, 168]
    assert len({row.video_id for row in rows}) == 1


def test_same_horizon_twice_corrects_the_reading(db: Session, niche) -> None:
    script = add_script(db, niche, [])
    save_performance(db, script, stats(100))

    save_performance(db, script, stats(200))

    row = db.scalars(select(Performance)).one()
    assert row.views == 20000


def test_late_entry_is_placed_in_time(db: Session, niche) -> None:
    history(db, niche)
    recent = add_script(db, niche, [])
    row = save_performance(db, recent, stats(100, days_ago=1))
    assert row.composite == pytest.approx(1.0)

    older = add_script(db, niche, [])
    save_performance(db, older, stats(100000, days_ago=50))

    db.refresh(row)
    assert row.baseline_size == 4


def test_invalid_stats_are_refused() -> None:
    with pytest.raises(ValidationError):
        PerformanceInput(horizon_h=72, retention_3s=45)
    with pytest.raises(ValidationError):
        PerformanceInput(horizon_h=48)
    with pytest.raises(ValidationError):
        PerformanceInput(horizon_h=72, views=-1)


def test_stats_link_the_script_to_a_video_and_publish_it(db: Session, niche) -> None:
    script = add_script(db, niche, [])

    save_performance(db, script, stats(video_url="https://tiktok.com/v/1"))

    video = db.get(Video, script.published_video_id)
    assert video.source_url == "https://tiktok.com/v/1"
    assert video.is_own_content is True
    assert video.niche_source == niche.name
    assert script.status is ScriptStatus.PUBLISHED


def test_already_ingested_video_is_reused(db: Session, niche) -> None:
    video = Video(source_url="https://tiktok.com/v/1")
    db.add(video)
    db.commit()
    script = add_script(db, niche, [])

    save_performance(db, script, stats(video_url="https://tiktok.com/v/1"))

    assert script.published_video_id == video.id


def test_linking_the_real_video_moves_the_readings(db: Session, niche) -> None:
    script = add_script(db, niche, [])
    row = save_performance(db, script, stats())
    real = Video(source_url="https://tiktok.com/vraie", is_own_content=True)
    db.add(real)
    db.commit()

    update_script(db, script, published_video_id=real.id)

    db.refresh(row)
    assert row.video_id == real.id


def test_success_adds_to_alpha_and_failure_to_beta(db: Session, niche) -> None:
    history(db, niche)
    good, bad = add_pattern(db, "good"), add_pattern(db, "bad")

    save_performance(db, add_script(db, niche, [good.id]), stats(200))
    save_performance(db, add_script(db, niche, [bad.id]), stats(50))

    db.refresh(good)
    db.refresh(bad)
    assert good.score_alpha == pytest.approx(2, abs=0.01)
    assert good.score_beta == 1
    assert bad.score_alpha == 1
    assert bad.score_beta == pytest.approx(2, abs=0.01)


def test_baseline_itself_is_not_a_success(db: Session, niche) -> None:
    history(db, niche)
    pattern = add_pattern(db)

    save_performance(db, add_script(db, niche, [pattern.id]), stats(100))

    db.refresh(pattern)
    assert pattern.score_alpha == 1
    assert pattern.score_beta == pytest.approx(2, abs=0.01)


def test_a_video_votes_once_even_with_several_readings(db: Session, niche) -> None:
    for horizon in (24, 72, 168):
        history(db, niche, horizon=horizon)
    pattern = add_pattern(db)
    script = add_script(db, niche, [pattern.id])

    for horizon in (24, 72, 168):
        save_performance(db, script, stats(200, horizon=horizon))

    db.refresh(pattern)
    assert pattern.score_alpha == pytest.approx(2, abs=0.01)


def test_longest_horizon_decides(db: Session, niche) -> None:
    for horizon in (24, 168):
        history(db, niche, horizon=horizon)
    pattern = add_pattern(db)
    script = add_script(db, niche, [pattern.id])

    save_performance(db, script, stats(500, horizon=24))
    save_performance(db, script, stats(10, horizon=168))

    db.refresh(pattern)
    assert pattern.score_beta > pattern.score_alpha


def test_correction_does_not_leave_a_false_trace(db: Session, niche) -> None:
    history(db, niche)
    pattern = add_pattern(db)
    script = add_script(db, niche, [pattern.id])

    save_performance(db, script, stats(500))
    save_performance(db, script, stats(10))

    db.refresh(pattern)
    assert pattern.score_alpha == 1
    assert pattern.score_beta == pytest.approx(2, abs=0.01)


def test_old_results_weigh_less(db: Session, niche) -> None:
    history(db, niche)
    pattern = add_pattern(db)
    save_performance(db, add_script(db, niche, [pattern.id]), stats(200, days_ago=90))

    rebuild_pattern_scores(db, NOW)
    db.commit()

    db.refresh(pattern)
    assert pattern.score_alpha == pytest.approx(1.5, abs=0.01)


def test_scores_are_separate_by_niche(db: Session, niche) -> None:
    other = make_niche(db)
    history(db, niche)
    pattern = add_pattern(db)

    save_performance(db, add_script(db, niche, [pattern.id]), stats(200))
    save_performance(db, add_script(db, other, [pattern.id]), stats(10))

    db.refresh(pattern)
    assert pattern.score_alpha == pytest.approx(2, abs=0.01)
    assert pattern.score_beta == pytest.approx(2, abs=0.01)
    assert niche_scores(db, niche.id)[pattern.id][0] == pytest.approx(2, abs=0.01)
    assert niche_scores(db, other.id)[pattern.id][1] == pytest.approx(2, abs=0.01)


def test_top_patterns_use_the_score_of_the_niche(db: Session, niche) -> None:
    other = make_niche(db)
    history(db, niche)
    a, b = add_pattern(db, "a"), add_pattern(db, "b")
    save_performance(db, add_script(db, niche, [a.id]), stats(10))
    save_performance(db, add_script(db, other, [a.id]), stats(500))
    save_performance(db, add_script(db, other, [a.id]), stats(500))
    save_performance(db, add_script(db, niche, [b.id]), stats(500))

    assert [item.name for item in top_patterns(db, niche_id=niche.id)] == ["b", "a"]


def test_post_mortem_waits_for_72_hours(db: Session, niche) -> None:
    history(db, niche, horizon=24)
    history(db, niche, horizon=72)
    script = add_script(db, niche, [])
    fake = FakeLearningClaude()

    assert record_performance(db, script, stats(horizon=24), fake).post_mortem is None
    result = record_performance(db, script, stats(horizon=72), fake)

    assert result.post_mortem.content["summary"] == "Résumé du diagnostic"
    assert fake.count("post_mortem") == 1


def test_no_post_mortem_without_a_score(db: Session, niche) -> None:
    fake = FakeLearningClaude()

    result = record_performance(db, add_script(db, niche, []), stats(), fake)

    assert result.post_mortem is None
    assert fake.calls == []


def test_post_mortem_receives_plan_reality_and_stats(db: Session, niche) -> None:
    history(db, niche)
    script = add_script(db, niche, [], predicted=73.0)
    fake = FakeLearningClaude()
    row = save_performance(db, script, stats(300, notes="Tourné dans le char"))
    db.add(Transcript(video_id=row.video_id, full_text="Ce que j'ai vraiment dit"))
    db.commit()

    record_performance(db, script, stats(300, notes="Tourné dans le char"), fake)

    variables = fake.calls[0]["variables"]
    assert "Le vrai problème" in variables["script"]
    assert variables["actual_transcript"] == "Ce que j'ai vraiment dit"
    assert '"predicted_score": 73.0' in variables["stats"]
    assert "winner" in variables["stats"]
    assert "Tourné dans le char" in variables["stats"]


def test_post_mortem_is_not_paid_twice(db: Session, niche) -> None:
    history(db, niche)
    script = add_script(db, niche, [])
    fake = FakeLearningClaude()
    record_performance(db, script, stats(), fake)

    record_performance(db, script, stats(), fake)

    assert fake.count("post_mortem") == 1


def test_failed_post_mortem_keeps_the_stats(db: Session, niche) -> None:
    history(db, niche)
    script = add_script(db, niche, [])

    result = record_performance(db, script, stats(), FakeLearningClaude(fail_on="post_mortem"))

    assert result.performance.composite is not None
    assert "Post-mortem non généré" in result.warnings[0]


def run_results(db: Session, niche, fake, count: int, pattern_ids=None, value=150) -> list[Script]:
    scripts = []
    for index in range(count):
        script = add_script(db, niche, pattern_ids or [])
        record_performance(db, script, stats(value, days_ago=count - index), fake)
        scripts.append(script)
    return scripts


def lesson(video_ids: list[int], **overrides: Any) -> dict[str, Any]:
    return {
        "statement": "Les hooks avec chiffre font mieux (5 vidéos)",
        "scope": "niche",
        "evidence_video_ids": video_ids,
        "confidence": 0.6,
        "avoid_pattern_ids": [],
    } | overrides


def test_lessons_are_rewritten_every_5_results(db: Session, niche) -> None:
    history(db, niche)
    fake = FakeLearningClaude()

    run_results(db, niche, fake, 4)
    assert fake.count("lessons_synthesis") == 0

    run_results(db, niche, fake, 1)
    assert fake.count("lessons_synthesis") == 1

    run_results(db, niche, fake, 4)
    assert fake.count("lessons_synthesis") == 1

    run_results(db, niche, fake, 1)
    assert fake.count("lessons_synthesis") == 2
    assert db.scalar(select(func.count()).select_from(SynthesisRun)) == 2


def test_synthesis_rereads_every_post_mortem(db: Session, niche) -> None:
    history(db, niche)
    fake = FakeLearningClaude()

    run_results(db, niche, fake, 5)

    call = next(c for c in fake.calls if c["prompt"] == "lessons_synthesis")
    assert call["variables"]["results"].count("Résumé du diagnostic") == 5


def video_ids(db: Session) -> list[int]:
    return list(db.scalars(select(PostMortem.video_id)))


def test_lesson_is_stored_with_its_proof(db: Session, niche) -> None:
    history(db, niche)
    run_results(db, niche, FakeLearningClaude(), 4)
    ids = video_ids(db)

    run_results(db, niche, FakeLearningClaude([lesson(ids)]), 1)

    stored = db.scalars(select(Lesson)).one()
    assert stored.active is True
    assert stored.scope is LessonScope.NICHE
    assert stored.niche_id == niche.id
    assert stored.confidence == 0.6
    assert [item["video_id"] for item in stored.evidence] == ids


def test_lesson_without_a_real_video_is_dropped(db: Session, niche) -> None:
    history(db, niche)

    run_results(db, niche, FakeLearningClaude([lesson([999])]), 5)

    assert db.scalars(select(Lesson)).all() == []


def test_invented_video_ids_are_removed_from_the_proof(db: Session, niche) -> None:
    history(db, niche)
    run_results(db, niche, FakeLearningClaude(), 4)
    ids = video_ids(db)

    run_results(db, niche, FakeLearningClaude([lesson([*ids, 999])]), 1)

    assert 999 not in [item["video_id"] for item in db.scalars(select(Lesson)).one().evidence]


def test_old_lessons_are_deactivated_never_deleted(db: Session, niche) -> None:
    old = Lesson(scope=LessonScope.NICHE, niche_id=niche.id, statement="Ancienne", evidence=[])
    replaced = Lesson(
        scope=LessonScope.NICHE, niche_id=niche.id, statement="Remplacée", evidence=[]
    )
    db.add_all([old, replaced])
    db.commit()
    history(db, niche)
    fake = FakeLearningClaude(
        contradicted=[{"lesson_id": old.id, "reason": "Les chiffres disent non"}]
    )

    run_results(db, niche, fake, 5)

    db.refresh(old)
    db.refresh(replaced)
    assert old.active is False
    assert old.deactivated_reason == "Les chiffres disent non"
    assert old.deactivated_at is not None
    assert replaced.active is False
    assert "Remplacée" in replaced.deactivated_reason
    assert db.scalar(select(func.count()).select_from(Lesson)) == 2


def test_one_bad_result_cannot_ban_a_pattern(db: Session, niche) -> None:
    history(db, niche)
    pattern = add_pattern(db)
    run_results(db, niche, FakeLearningClaude(), 4)
    script = add_script(db, niche, [pattern.id])
    ids = video_ids(db)
    fake = FakeLearningClaude([lesson(ids, avoid_pattern_ids=[pattern.id])])

    record_performance(db, script, stats(10), fake)

    assert all("effect" not in item for item in db.scalars(select(Lesson)).one().evidence)


def test_repeated_failures_can_ban_a_pattern(db: Session, niche) -> None:
    history(db, niche)
    pattern = add_pattern(db)
    run_results(db, niche, FakeLearningClaude(), 4, [pattern.id], value=10)
    ids = video_ids(db)
    fake = FakeLearningClaude([lesson(ids, avoid_pattern_ids=[pattern.id])])

    run_results(db, niche, fake, 1, [pattern.id], value=10)

    assert {"pattern_id": pattern.id, "effect": "avoid"} in db.scalars(
        select(Lesson)
    ).one().evidence


def test_failed_synthesis_keeps_results_and_active_lessons(db: Session, niche) -> None:
    kept = Lesson(scope=LessonScope.NICHE, niche_id=niche.id, statement="Gardée", evidence=[])
    db.add(kept)
    db.commit()
    history(db, niche)
    run_results(db, niche, FakeLearningClaude(), 4)
    script = add_script(db, niche, [])

    result = record_performance(
        db, script, stats(), FakeLearningClaude(fail_on="lessons_synthesis")
    )

    assert "Leçons non mises à jour" in result.warnings[0]
    db.refresh(kept)
    assert kept.active is True


def test_synthesis_retries_after_a_failure(db: Session, niche) -> None:
    history(db, niche)
    run_results(db, niche, FakeLearningClaude(fail_on="lessons_synthesis"), 5)

    assert maybe_synthesize(db, niche, FakeLearningClaude()) is True


def test_force_rewrites_without_waiting(db: Session, niche) -> None:
    history(db, niche)
    run_results(db, niche, FakeLearningClaude(), 1)
    fake = FakeLearningClaude()

    assert maybe_synthesize(db, niche, fake) is False
    assert maybe_synthesize(db, niche, fake, force=True) is True


def test_active_lessons_reach_the_critic(db: Session, niche) -> None:
    from app.generate.guards import ScriptGuard
    from app.generate.schemas import ScriptDraft
    from app.generate.service import Brief, _polish

    fake = FakeScriptClaude(scores=[9])
    brief = Brief(
        mode="generate",
        niche=niche,
        profile=load_profile(niche),
        pattern_ids=[],
        topic_ids=[],
        content={"lessons": ["Le hook doit nommer l'âge"]},
        guard=ScriptGuard(),
        words_per_second=3,
        duration_source="default",
    )

    _polish(fake, brief, ScriptDraft.model_validate(script_draft([])))

    assert "Le hook doit nommer l'âge" in fake.calls[0]["variables"]["lessons"]


def test_exploration_takes_the_least_tested_pattern() -> None:
    old = Pattern(id=1, type="hook", name="old", score_alpha=9.0, score_beta=4.0)
    new = Pattern(id=2, type="hook", name="new", score_alpha=1.0, score_beta=1.0)

    assert exploration_combination([old, new], np.random.default_rng(0))[0].name == "new"


def test_exploration_uses_the_scores_of_the_niche() -> None:
    a = Pattern(id=1, type="hook", name="a", score_alpha=1.0, score_beta=1.0)
    b = Pattern(id=2, type="hook", name="b", score_alpha=3.0, score_beta=1.0)

    chosen = exploration_combination([a, b], np.random.default_rng(0), {1: (9.0, 9.0)})

    assert chosen[0].name == "b"


def test_exploration_slots_average_20_percent() -> None:
    rng = np.random.default_rng(3)

    slots = [exploration_slots(3, 0.2, rng) for _ in range(5000)]

    assert np.mean(slots) == pytest.approx(0.6, abs=0.03)
    assert exploration_slots(5, 0.2, np.random.default_rng(0)) == 1


class PatternAwareScripts(FakeScriptClaude):
    def _reply(self, prompt, schema, variables):
        import json

        if prompt == "script_write":
            chosen = [item["id"] for item in json.loads(variables["brief"])["patterns"]]
            return schema.model_validate(script_draft(chosen))
        return super()._reply(prompt, schema, variables)


def test_generate_reserves_and_flags_exploration(db: Session, niche, monkeypatch) -> None:
    monkeypatch.setenv("LEARNING_EXPLORATION_RATE", "1.0")
    get_settings.cache_clear()
    add_pattern(db, "H1")
    add_pattern(db, "H2")

    result = generate(db, niche.id, PatternAwareScripts(), n=4, develop=2, seed=1)

    assert all(load_content(script).experiment for script in result.scripts)
    assert "**TEST :**" in render_script(load_content(result.scripts[0]))


def test_generate_without_exploration_flags_nothing(db: Session, niche, monkeypatch) -> None:
    monkeypatch.setenv("LEARNING_EXPLORATION_RATE", "0")
    get_settings.cache_clear()
    add_pattern(db)

    result = generate(db, niche.id, PatternAwareScripts(), n=2, develop=1, seed=1)

    assert load_content(result.scripts[0]).experiment is False


def test_idea_for_exploration_beats_better_ideas_when_a_slot_is_reserved(
    db: Session, niche, monkeypatch
) -> None:
    monkeypatch.setenv("LEARNING_EXPLORATION_RATE", "0.5")
    get_settings.cache_clear()
    add_pattern(db)

    class LowTestPotential(PatternAwareScripts):
        def _reply(self, prompt, schema, variables):
            if prompt == "script_ideas":
                import json

                topics = json.loads(variables["topics"])
                count = variables["count"]
                return schema.model_validate(
                    {
                        "ideas": [
                            {
                                "combination_index": index,
                                "topic_id": topics[index]["id"],
                                "core_idea": f"c{index}",
                                "angle": "a",
                                "belief_attacked": "b",
                                "why": "w",
                                "potential": count - index,
                            }
                            for index in range(count)
                        ]
                    }
                )
            return super()._reply(prompt, schema, variables)

    result = generate(db, niche.id, LowTestPotential(), n=4, develop=2, seed=1)

    assert sum(load_content(item).experiment for item in result.scripts) >= 1


def test_prediction_accuracy() -> None:
    assert prediction_accuracy([(1, 1)])["correlation"] is None

    perfect = prediction_accuracy([(40, 0.5), (50, 0.8), (60, 1.2), (70, 1.6)])

    assert perfect["sample"] == 4
    assert perfect["correlation"] > 0.9
    assert perfect["direction_accuracy"] == 1


def test_prediction_accuracy_without_variance_gives_no_correlation() -> None:
    assert prediction_accuracy([(50, 1.0), (50, 1.2), (50, 0.8)])["correlation"] is None


def test_report(db: Session, niche) -> None:
    history(db, niche)
    good, bad = add_pattern(db, "good"), add_pattern(db, "bad")
    save_performance(db, add_script(db, niche, [good.id], 70), stats(500))
    save_performance(db, add_script(db, niche, [bad.id], 40), stats(10))
    save_performance(db, add_script(db, niche, [bad.id], 30), stats(20))
    db.add(
        Lesson(
            scope=LessonScope.NICHE,
            niche_id=niche.id,
            statement="Une leçon",
            evidence=[{"video_id": 1}],
        )
    )
    db.commit()

    report = build_report(db, niche.id)

    assert report["scored_videos"] == 3
    assert report["labels"]["winner"] == 1
    assert report["composite_by_month"][0]["videos"] == 3
    assert report["patterns"]["top"][0]["name"] == "good"
    assert report["patterns"]["flop"][0]["name"] == "bad"
    assert report["prediction"]["sample"] == 3
    assert report["active_lessons"][0]["statement"] == "Une leçon"
    assert report["active_lessons"][0]["videos"] == [1]


def test_report_on_an_empty_database(db: Session) -> None:
    report = build_report(db)

    assert report["scored_videos"] == 0
    assert report["prediction"]["correlation"] is None


@pytest.mark.parametrize(
    ("raw", "expected"), [("45%", 0.45), ("45 %", 0.45), ("0,45", 0.45), ("45", 0.45), ("1", 1.0)]
)
def test_parse_rate(raw: str, expected: float) -> None:
    assert parse_rate(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("raw", "expected"), [("1:05", 65), ("0:01:05", 65), ("12,4", 12.4), ("12 s", 12)]
)
def test_parse_seconds(raw: str, expected: float) -> None:
    assert parse_seconds(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("12 345", 12345), ("12,345", 12345), ("1.2K", 1200), ("3M", 3000000), ("7", 7)],
)
def test_parse_count(raw: str, expected: int) -> None:
    assert parse_count(raw) == expected


def test_csv_import_with_a_preset_and_a_script_id(db: Session, niche) -> None:
    script = add_script(db, niche, [])
    content = (
        f'script_id,Video views,Likes,Shares,Average watch time\n{script.id},"1,200",50,-,0:12\n'
    )

    result = import_csv(db, content, 72, "tiktok")

    assert result.imported == 1
    row = db.scalars(select(Performance)).one()
    assert (row.views, row.likes, row.shares, row.avg_watch_time_s) == (1200, 50, None, 12)
    assert row.horizon_h == 72


def test_csv_matches_rows_by_video_link(db: Session, niche) -> None:
    script = add_script(db, niche, [])
    save_performance(db, script, stats(video_url="https://t.co/1", horizon=24))
    content = "Lien;Vues\nhttps://t.co/1;500\nhttps://t.co/inconnu;9\n"

    result = import_csv(db, content, 72, mapping={"Lien": "video_url", "Vues": "views"})

    assert result.imported == 1
    assert "Ligne 3" in result.skipped[0]


def test_csv_bad_line_is_reported_and_the_rest_imported(db: Session, niche) -> None:
    script = add_script(db, niche, [])
    other = add_script(db, niche, [])
    content = f"script_id,views\n{script.id},pasunnombre\n{other.id},100\n"

    result = import_csv(db, content, 72, mapping={"views": "views"})

    assert result.imported == 1
    assert "Ligne 2" in result.skipped[0]


def test_csv_without_any_link_column_is_refused(db: Session) -> None:
    with pytest.raises(ImportError_, match="relier"):
        import_csv(db, "Vues\n100\n", 72, mapping={"Vues": "views"})


def test_csv_invalid_horizon_is_refused(db: Session) -> None:
    with pytest.raises(ImportError_, match="Horizon"):
        import_csv(db, "script_id\n1\n", 48)


def test_baseline_rises_as_a_winner_dominates(db: Session, niche) -> None:
    """Refaire le même niveau devient neutre : il faut progresser pour rester au-dessus."""
    history(db, niche, count=5, value=100)
    first = save_performance(db, add_script(db, niche, []), stats(300, days_ago=20))
    for index in range(5):
        save_performance(db, add_script(db, niche, []), stats(300, days_ago=19 - index))

    last = save_performance(db, add_script(db, niche, []), stats(300, days_ago=1))

    assert first.composite == pytest.approx(3.0)
    assert last.composite == pytest.approx(1.0)


def test_csv_unknown_preset_or_field_is_refused(db: Session) -> None:
    with pytest.raises(ImportError_, match="Préréglage"):
        import_csv(db, "a\n1\n", 72, "snapchat")
    with pytest.raises(ImportError_, match="Champs inconnus"):
        import_csv(db, "a\n1\n", 72, mapping={"a": "magie"})


client = TestClient(app)


@pytest.fixture
def api_claude():
    fake = FakeLearningClaude()
    app.dependency_overrides[get_claude] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


def test_performance_route(db: Session, niche, api_claude) -> None:
    history(db, niche)
    script = add_script(db, niche, [])

    response = client.post(
        f"/scripts/{script.id}/performance",
        json={"horizon_h": 72, "views": 30000, "shares": 300, "retention_3s": 0.3},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["composite"] == pytest.approx(3.0)
    assert body["label"] == "winner"
    assert body["post_mortem"]["summary"] == "Résumé du diagnostic"
    assert body["warnings"] == []


def test_performance_route_errors(db: Session, niche, api_claude) -> None:
    script = add_script(db, niche, [])

    assert client.post("/scripts/999/performance", json={"horizon_h": 72}).status_code == 404
    bad = client.post(
        f"/scripts/{script.id}/performance", json={"horizon_h": 72, "retention_3s": 45}
    )
    assert bad.status_code == 422


def test_import_route(db: Session, niche) -> None:
    script = add_script(db, niche, [])

    response = client.post(
        "/performance/import",
        files={"file": ("stats.csv", f"script_id,Views\n{script.id},100\n".encode())},
        data={"horizon_h": "72", "preset": "youtube"},
    )

    assert response.json() == {"imported": 1, "skipped": []}
    bad = client.post(
        "/performance/import",
        files={"file": ("stats.csv", b"Vues\n100\n")},
        data={"horizon_h": "72"},
    )
    assert bad.status_code == 422


def test_report_route(db: Session, niche) -> None:
    assert client.get(f"/learning/report?niche_id={niche.id}").json()["scored_videos"] == 0
    assert client.get("/learning/report?niche_id=999").status_code == 404


def test_form_is_served() -> None:
    response = client.get("/forms/performance")

    assert response.status_code == 200
    assert "Saisie des stats" in response.text


runner = CliRunner()


def test_cli_stats(db: Session, niche, monkeypatch: pytest.MonkeyPatch) -> None:
    history(db, niche)
    script = add_script(db, niche, [])
    monkeypatch.setattr(cli_module, "default_claude", FakeLearningClaude)
    answers = ["72", "https://t.co/9", "30000", "", "30", "", "", "0:12", "300", "", "", "", ""]

    result = runner.invoke(cli, ["stats", str(script.id)], input="\n".join(answers) + "\n")

    assert result.exit_code == 0, result.output
    assert "winner" in result.output
    assert "Résumé du diagnostic" in result.output
    row = db.scalars(select(Performance).where(Performance.script_id == script.id)).one()
    assert row.retention_3s == 0.3
    assert row.avg_watch_time_s == 12
    assert row.likes is None


def test_cli_stats_without_history_says_so(db: Session, niche, monkeypatch) -> None:
    script = add_script(db, niche, [])
    monkeypatch.setattr(cli_module, "default_claude", FakeLearningClaude)

    result = runner.invoke(cli, ["stats", str(script.id)], input="72\n\n100\n" + "\n" * 10)

    assert result.exit_code == 0, result.output
    assert "Pas encore de score" in result.output


def test_cli_stats_retries_an_unreadable_value(db: Session, niche, monkeypatch) -> None:
    script = add_script(db, niche, [])
    monkeypatch.setattr(cli_module, "default_claude", FakeLearningClaude)

    result = runner.invoke(cli, ["stats", str(script.id)], input="72\n\nabc\n100\n" + "\n" * 10)

    assert result.exit_code == 0, result.output
    assert db.scalars(select(Performance)).one().views == 100


def test_cli_unknown_script(db: Session) -> None:
    assert runner.invoke(cli, ["stats", "999"]).exit_code == 1


def test_cli_import_then_learn_then_report(db: Session, niche, tmp_path, monkeypatch) -> None:
    history(db, niche)
    fake = FakeLearningClaude()
    monkeypatch.setattr(cli_module, "default_claude", lambda: fake)
    script = add_script(db, niche, [])
    path = tmp_path / "stats.csv"
    path.write_text(f"script_id,Vues\n{script.id},30000\n", encoding="utf-8")

    imported = runner.invoke(
        cli, ["stats-import", str(path), "--horizon", "72", "--map", "Vues=views"]
    )
    assert imported.exit_code == 0, imported.output
    assert "1 ligne" in imported.output
    assert fake.calls == []

    assert runner.invoke(cli, ["learn"]).exit_code == 0
    assert fake.count("post_mortem") == 1

    assert '"scored_videos": 1' in runner.invoke(cli, ["report"]).output


@pytest.mark.parametrize(
    ("name", "variables"),
    [
        (
            "post_mortem",
            {"stats": "{}", "script": "{}", "actual_transcript": "", "published_analysis": ""},
        ),
        ("lessons_synthesis", {"results": "[]", "patterns": "[]", "lessons": "[]"}),
        ("script_critic", {"avatar": "{}", "script": "{}", "lessons": "[]"}),
    ],
)
def test_prompts_render(name: str, variables: dict) -> None:
    assert "{{" not in load_prompt(name, "v1").render_user(variables)


def test_critic_prompt_mentions_the_lessons() -> None:
    text = load_prompt("script_critic", "v1").render_user(
        {"avatar": "{}", "script": "{}", "lessons": '["Ma leçon"]'}
    )

    assert "Ma leçon" in text


def test_simulation_30_videos_with_one_clear_winner(db: Session, niche) -> None:
    """Le pattern gagnant voit son score monter, et il est choisi de plus en plus souvent."""
    rng = np.random.default_rng(42)
    winner = add_pattern(db, "gagnant")
    others = [add_pattern(db, f"moyen {index}") for index in range(3)]
    candidates = [winner, *others]
    history(db, niche, count=5, value=100)

    picks = []
    for index in range(30):
        scores = niche_scores(db, niche.id)
        chosen = thompson_combination(candidates, rng, scores)[0]
        picks.append(chosen.name)
        base = 220 if chosen.id == winner.id else 95
        value = float(base * rng.lognormal(0, 0.15))
        script = add_script(db, niche, [chosen.id])
        save_performance(db, script, stats(value, days_ago=30 - index))

    for pattern in candidates:
        db.refresh(pattern)

    # Le score absolu plafonne : plus le gagnant domine, plus la baseline monte vers son niveau.
    # C'est l'écart avec les autres patterns qui prouve que la machine a appris.
    assert score_mean(winner) > 0.6
    assert score_mean(winner) > max(score_mean(item) for item in others) + 0.2
    assert winner.score_alpha > 5

    first, last = picks[:10], picks[20:]
    assert last.count("gagnant") > first.count("gagnant")
    assert last.count("gagnant") >= 6

    final = niche_scores(db, niche.id)
    draws = [thompson_combination(candidates, rng, final)[0].name for _ in range(1000)]
    assert draws.count("gagnant") > 600

    report = build_report(db, niche.id)
    assert report["patterns"]["top"][0]["name"] == "gagnant"
    assert report["labels"]["winner"] >= 5


def test_stored_pattern_scores_are_replaced_not_stacked(db: Session, niche) -> None:
    history(db, niche)
    pattern = add_pattern(db)
    script = add_script(db, niche, [pattern.id])
    save_performance(db, script, stats(200))

    rebuild_pattern_scores(db)
    rebuild_pattern_scores(db)
    db.commit()

    assert db.scalar(select(func.count()).select_from(PatternScore)) == 1
    assert len(scored_results(db)) == 1
    assert active_lessons(db, niche.id) == []
