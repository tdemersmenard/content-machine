"""Boucle d'outils, outils, mémoire, budget, répétitions et API de l'agent."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import anthropic
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app import cli as cli_module
from app.agent.budget import (
    BudgetExceeded,
    budget_status,
    can_spend,
    check_budget,
    day_start,
    tokens_used,
)
from app.agent.chat import AgentError, chat, check_citations, start_session
from app.agent.memory import RECENT_SCRIPTS, load_memory
from app.agent.tools import MAX_RESULT_CHARS, TOOLS, ToolContext, api_tools, run_tool
from app.cli import cli
from app.config import get_settings
from app.db.models import (
    ActionStatus,
    AgentAction,
    AgentSession,
    Lesson,
    LessonScope,
    Niche,
    Notification,
    Script,
    ScriptStatus,
    WeeklyReport,
)
from app.generate.novelty import NoveltyFilter, idea_text
from app.generate.service import GenerationError, adapt, generate, load_content, update_script
from app.ingest.jobs import ingest
from app.learning.service import save_performance
from app.llm.client import get_claude
from app.llm.errors import ClaudeCredentialsError, ClaudeRefusalError
from app.llm.prompts import load_prompt
from app.main import app
from tests.agent_helpers import FakeAgentClaude, spend, text_turn, tool_turn
from tests.analysis_helpers import sample_analysis
from tests.ingest_helpers import make_deps
from tests.pattern_helpers import FakeEmbedder
from tests.script_helpers import script_draft
from tests.test_learning import PatternAwareScripts, add_pattern, add_script, history, stats
from tests.test_niche import make_niche

EXPECTED_TOOLS = {
    "ingest",
    "analyze",
    "search_patterns",
    "top_patterns",
    "get_niche",
    "adapt",
    "generate",
    "get_performance",
    "run_post_mortem",
    "synthesize_lessons",
    "learning_report",
}


@pytest.fixture
def niche(db: Session) -> Niche:
    return make_niche(db)


def context(db: Session, niche: Niche, claude=None, **kwargs) -> ToolContext:
    return ToolContext(session=db, niche=niche, claude=claude or FakeAgentClaude(), **kwargs)


def talk(db: Session, niche: Niche, fake: FakeAgentClaude, message: str = "Salut", **kwargs):
    return chat(db, start_session(db, niche.id), message, fake, **kwargs)


def other_niche(db: Session) -> Niche:
    row = Niche(name="Autre")
    db.add(row)
    db.commit()
    return row


def last_action(db: Session, name: str) -> AgentAction:
    return db.scalar(
        select(AgentAction)
        .where(AgentAction.action == name)
        .order_by(AgentAction.id.desc())
        .limit(1)
    )


def test_the_agent_has_exactly_the_requested_tools() -> None:
    assert set(TOOLS) == EXPECTED_TOOLS


def test_no_tool_can_publish() -> None:
    """Le créateur reste le seul à publier : rien n'y ressemble, même dans les descriptions."""
    for tool in api_tools():
        assert "publish" not in tool["name"]
        assert "post_to" not in tool["name"]
        assert "publier" not in tool["description"].lower()


def test_every_tool_is_described_and_has_an_object_schema() -> None:
    tools = api_tools()

    assert {item["name"] for item in tools} == EXPECTED_TOOLS
    for tool in tools:
        assert len(tool["description"]) > 20
        assert tool["input_schema"]["type"] == "object"
        assert tool["eager_input_streaming"] is True


def test_free_tools_cost_nothing_and_paid_tools_estimate(db: Session, niche: Niche) -> None:
    ctx = context(db, niche)
    free = ("search_patterns", "top_patterns", "get_niche", "get_performance", "learning_report")
    for name in free:
        tool = TOOLS[name]
        assert tool.tokens(ctx, None) == 0
    for name in ("ingest", "adapt", "generate", "run_post_mortem", "synthesize_lessons"):
        tool = TOOLS[name]
        data = tool.input_model.model_validate(
            {"ingest": {"source": "x"}, "adapt": {"video_id": 1}}.get(name, {})
            | ({"script_id": 1} if name == "run_post_mortem" else {})
        )
        assert tool.tokens(ctx, data) > 0


def test_unknown_tool_is_refused_and_logged(db: Session, niche: Niche) -> None:
    outcome = run_tool(context(db, niche), "publish_video", {})

    assert outcome.is_error
    assert last_action(db, "publish_video").status is ActionStatus.REFUSED


def test_invalid_input_goes_back_to_claude(db: Session, niche: Niche) -> None:
    outcome = run_tool(context(db, niche), "generate", {"count": 99})

    assert outcome.is_error
    assert "count" in outcome.content
    assert last_action(db, "generate").status is ActionStatus.REFUSED


def test_extra_input_is_refused(db: Session, niche: Niche) -> None:
    assert run_tool(context(db, niche), "get_niche", {"hack": True}).is_error


def test_paid_tool_is_refused_over_budget_before_any_call(
    db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_DAILY_TOKEN_BUDGET", "1000")
    get_settings.cache_clear()
    fake = FakeAgentClaude()

    outcome = run_tool(context(db, niche, fake), "generate", {"count": 1})

    assert outcome.is_error
    assert "Budget" in outcome.content
    assert fake.calls == []
    assert last_action(db, "generate").status is ActionStatus.REFUSED


def test_free_tool_still_works_over_budget(
    db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_DAILY_TOKEN_BUDGET", "0")
    get_settings.cache_clear()
    spend(db, 500)

    assert not run_tool(context(db, niche), "learning_report", {}).is_error


def test_tool_failure_is_logged_and_never_raises(
    db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.agent.tools.build_report", lambda *_: 1 / 0)

    outcome = run_tool(context(db, niche), "learning_report", {})

    assert outcome.is_error
    row = last_action(db, "learning_report")
    assert row.status is ActionStatus.ERROR
    assert "ZeroDivisionError" in row.summary
    assert row.finished_at is not None


def test_long_results_are_cut(db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.agent.tools.build_report", lambda *_: {"x": "a" * 50_000})

    outcome = run_tool(context(db, niche), "learning_report", {})

    assert len(outcome.content) < MAX_RESULT_CHARS + 100
    assert "tronqué" in outcome.content


def test_ingest_refuses_a_file_that_is_not_a_video(db: Session, niche: Niche) -> None:
    outcome = run_tool(context(db, niche), "ingest", {"source": "C:/secrets/.env"})

    assert outcome.is_error
    assert "vidéo" in outcome.content


def test_ingest_and_analyze_tools_on_a_real_file(
    db: Session, niche: Niche, sample_video: Path
) -> None:
    from tests.analysis_helpers import FakeAnalysisClaude

    fake = FakeAnalysisClaude()
    ctx = context(db, niche, fake, ingest_deps=make_deps(), embedder=FakeEmbedder())

    ingested = run_tool(ctx, "ingest", {"source": str(sample_video)})
    video_id = json.loads(ingested.content)["video_id"]
    again = json.loads(run_tool(ctx, "ingest", {"source": str(sample_video)}).content)
    analysis = run_tool(ctx, "analyze", {"video_id": video_id})

    assert not ingested.is_error
    assert again["already_known"] is True
    assert not analysis.is_error
    digest = json.loads(analysis.content)
    assert digest["video_id"] == video_id
    assert digest["why_it_worked"][0]["rank"] == 1
    assert digest["hook"]["type"] == sample_analysis()["hook"]["hook_type"]


def test_get_performance_defaults_to_the_last_video(db: Session, niche: Niche) -> None:
    history(db, niche)
    script = add_script(db, niche, [])
    save_performance(db, script, stats(10, days_ago=1))

    result = json.loads(run_tool(context(db, niche), "get_performance", {}).content)

    assert result["script_id"] == script.id
    assert result["readings"][0]["label"] == "loser"


def test_get_performance_without_stats_says_so(db: Session, niche: Niche) -> None:
    result = json.loads(run_tool(context(db, niche), "get_performance", {}).content)

    assert "Aucune stat" in result["message"]


def test_script_of_another_niche_is_refused(db: Session, niche: Niche) -> None:
    other = other_niche(db)
    script = add_script(db, other, [])

    outcome = run_tool(context(db, niche), "get_performance", {"script_id": script.id})

    assert outcome.is_error


def test_post_mortem_tool_explains_why_it_cannot_run(db: Session, niche: Niche) -> None:
    script = add_script(db, niche, [])

    outcome = run_tool(context(db, niche), "run_post_mortem", {"script_id": script.id})

    assert outcome.is_error
    assert "baseline" in outcome.content


def test_chat_uses_a_tool_then_answers(db: Session, niche: Niche) -> None:
    fake = FakeAgentClaude(
        [
            tool_turn("learning_report", say="Je regarde ton rapport."),
            text_turn("Rien de mesuré pour l'instant."),
        ]
    )
    steps = []

    result = talk(db, niche, fake, on_step=steps.append)

    assert result.answer == "Rien de mesuré pour l'instant."
    assert [item.kind for item in steps] == ["text", "tool_start", "tool_end"]
    assert steps[1].name == "learning_report"
    assert not steps[2].is_error
    second = fake.tool_calls[1]["messages"]
    assert [item["role"] for item in second] == ["user", "assistant", "user"]
    reply = second[-1]["content"][0]
    assert reply["type"] == "tool_result"
    assert reply["tool_use_id"] == "toolu_learning_report"
    assert "is_error" not in reply


def test_tool_error_is_flagged_for_claude(db: Session, niche: Niche) -> None:
    fake = FakeAgentClaude([tool_turn("analyze", {"video_id": 999}), text_turn("Vidéo inconnue.")])

    result = talk(db, niche, fake)

    assert result.steps[-1].is_error
    assert fake.tool_calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_first_message_carries_the_memory_and_later_ones_do_not(db: Session, niche: Niche) -> None:
    fake = FakeAgentClaude([text_turn("Un"), text_turn("Deux")])
    agent_session = start_session(db, niche.id)

    chat(db, agent_session, "Premier", fake)
    chat(db, agent_session, "Deuxième", fake)

    first = fake.tool_calls[0]["messages"][0]["content"]
    assert "Mémoire au début" in first[0]["text"]
    assert niche.name in first[0]["text"]
    assert first[1]["text"] == "Premier"
    assert fake.tool_calls[1]["messages"][-1] == {"role": "user", "content": "Deuxième"}
    assert len(fake.tool_calls[1]["messages"]) == 3


def test_history_is_saved_and_json_serializable(db: Session, niche: Niche) -> None:
    fake = FakeAgentClaude([tool_turn("get_niche"), text_turn("Voilà.")])

    result = talk(db, niche, fake)

    db.expire_all()
    messages = db.get(AgentSession, result.session_id).messages
    assert json.dumps(messages)
    assert [item["role"] for item in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[1]["content"][0]["type"] == "tool_use"


def test_too_many_rounds_stop_cleanly(
    db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_MAX_TOOL_ROUNDS", "2")
    get_settings.cache_clear()
    fake = FakeAgentClaude([tool_turn("get_niche"), tool_turn("get_niche")])

    result = talk(db, niche, fake)

    assert "2 étapes" in result.interrupted
    db.expire_all()
    assert db.get(AgentSession, result.session_id).messages[-1]["role"] == "assistant"


def test_budget_stops_the_turn_before_calling_claude(
    db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_DAILY_TOKEN_BUDGET", "100")
    get_settings.cache_clear()
    fake = FakeAgentClaude()

    result = talk(db, niche, fake)

    assert "Budget" in result.interrupted
    assert fake.tool_calls == []
    assert last_action(db, "chat_turn").status is ActionStatus.REFUSED


def test_conversation_stays_valid_after_a_refusal(db: Session, niche: Niche) -> None:
    fake = FakeAgentClaude(
        [tool_turn("get_niche"), ClaudeRefusalError("Refus", category="x"), text_turn("Reprise")]
    )
    agent_session = start_session(db, niche.id)

    interrupted = chat(db, agent_session, "Un", fake)
    resumed = chat(db, agent_session, "Deux", fake)

    assert "Refus" in interrupted.interrupted
    assert resumed.answer == "Reprise"
    roles = [item["role"] for item in fake.tool_calls[-1]["messages"]]
    assert all(a != b for a, b in zip(roles, roles[1:], strict=False))


def test_api_failure_does_not_lose_the_scripts_already_paid(db: Session, niche: Niche) -> None:
    add_pattern(db)
    request = anthropic.APIConnectionError(request=None)  # type: ignore[arg-type]
    fake = FakeAgentClaude([request])

    result = talk(db, niche, fake)

    assert "Échec" in result.interrupted


def test_missing_key_is_not_swallowed(db: Session, niche: Niche) -> None:
    fake = FakeAgentClaude([ClaudeCredentialsError("Aucune clé")])

    with pytest.raises(ClaudeCredentialsError):
        talk(db, niche, fake)


def test_citations_are_checked_against_the_database(db: Session, niche: Niche) -> None:
    pattern = add_pattern(db)
    script = add_script(db, niche, [])
    text = f"[pattern {pattern.id}] et [script {script.id}], [Vidéo 404], [leçon 9]"

    evidence, unverified = check_citations(db, text)

    assert evidence == {"pattern": [pattern.id], "script": [script.id]}
    assert unverified == ["video 404", "lesson 9"]


def test_chat_reports_invented_references(db: Session, niche: Niche) -> None:
    fake = FakeAgentClaude([text_turn("D'après [vidéo 77], ça marche.")])

    result = talk(db, niche, fake)

    assert result.unverified == ["video 77"]


def test_session_needs_a_niche_with_a_profile(db: Session) -> None:
    with pytest.raises(AgentError):
        start_session(db, 999)


def test_every_turn_is_logged(db: Session, niche: Niche) -> None:
    fake = FakeAgentClaude([tool_turn("get_niche"), text_turn("Fini.")])

    result = talk(db, niche, fake, "Qui suis-je ?")

    actions = db.scalars(select(AgentAction).order_by(AgentAction.id)).all()
    assert [item.action for item in actions] == ["chat_turn", "get_niche"]
    assert all(item.session_id == result.session_id for item in actions)
    assert actions[0].input == {"message": "Qui suis-je ?"}
    assert actions[0].summary == "Fini."
    assert all(item.status is ActionStatus.OK for item in actions)


def test_generate_through_the_agent_writes_scripts(db: Session, niche: Niche) -> None:
    add_pattern(db)
    fake = FakeAgentClaude(
        [tool_turn("generate", {"count": 2}), text_turn("Deux scripts.")],
        drafts=[script_draft([1], angle=f"Angle {index}") for index in range(2)],
    )

    result = talk(db, niche, fake, "Fais-moi 2 scripts")

    output = json.loads(result.steps[-1].detail)
    assert len(output["scripts"]) == 2
    assert db.scalar(select(func.count()).select_from(Script)) == 2
    assert all(item.status is ScriptStatus.DRAFT for item in db.scalars(select(Script)))


def test_memory_has_lessons_report_and_last_ten_scripts(db: Session, niche: Niche) -> None:
    scripts = [add_script(db, niche, []) for _ in range(RECENT_SCRIPTS + 2)]
    update_script(db, scripts[-1], status=ScriptStatus.APPROVED)
    history(db, niche, count=3)
    save_performance(db, scripts[-2], stats(500, days_ago=1))
    db.add(
        Lesson(
            scope=LessonScope.NICHE, niche_id=niche.id, statement="Court", evidence=[], confidence=1
        )
    )
    db.add(WeeklyReport(niche_id=niche.id, week_start="2026-11-02", markdown="# Rapport"))
    db.add(WeeklyReport(niche_id=niche.id, week_start="2026-10-26", markdown="# Ancien"))
    db.commit()

    memory = load_memory(db, niche)

    assert memory["niche"]["profile"] == niche.profile
    assert [item["statement"] for item in memory["active_lessons"]] == ["Court"]
    assert memory["latest_weekly_report"] == "# Rapport"
    assert memory["learning_report"]["scored_videos"] >= 1
    assert len(memory["recent_scripts"]) == RECENT_SCRIPTS
    by_id = {item["id"]: item for item in memory["recent_scripts"]}
    assert scripts[0].id not in by_id or len(by_id) == RECENT_SCRIPTS
    assert by_id[scripts[-2].id]["result"]["label"] == "winner"
    assert by_id[scripts[-2].id]["status"] == "published"


def test_budget_counts_billed_tokens_but_not_cache_reads(db: Session) -> None:
    spend(db, 1000, cache_read=50_000)

    assert tokens_used(db) == 1000


def test_budget_resets_at_local_midnight(db: Session) -> None:
    # 3 h 30 UTC le 6 novembre : il est encore le 5 au soir à Montréal.
    now = datetime(2026, 11, 6, 3, 30, tzinfo=UTC)
    spend(db, 100, created_at=datetime(2026, 11, 6, 1, 0, tzinfo=UTC))
    spend(db, 200, created_at=datetime(2026, 11, 5, 3, 0, tzinfo=UTC))

    assert day_start(now) == datetime(2026, 11, 5, 5, 0, tzinfo=UTC)
    assert tokens_used(db, now) == 100


def test_budget_refuses_what_does_not_fit(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_DAILY_TOKEN_BUDGET", "1000")
    get_settings.cache_clear()
    spend(db, 900)

    assert can_spend(db, 100)
    assert not can_spend(db, 101)
    with pytest.raises(BudgetExceeded, match="900"):
        check_budget(db, 200)
    assert budget_status(db) == {"limit": 1000, "used": 900, "remaining": 100, "cost_usd": 0.5}


def novelty_niche_script(
    db: Session, niche: Niche, angle: str, belief: str, core_idea: str = ""
) -> Script:
    script = add_script(db, niche, [])
    content = dict(script.content, angle=angle, belief_attacked=belief, core_idea=core_idea)
    script.content = content
    db.commit()
    return script


def test_filter_catches_a_repeat_and_keeps_a_new_idea(db: Session, niche: Niche) -> None:
    known = novelty_niche_script(db, niche, "Garder l'école et partir un service", "Lâcher l'école")
    novelty = NoveltyFilter(db, niche.id, FakeEmbedder())

    repeat, fresh = novelty.check(
        [
            idea_text("Garder l'école et partir un service", "Lâcher l'école"),
            idea_text("Répondre à un client trop pressé", "Baisser ses prix"),
        ]
    )

    assert repeat.script_id == known.id
    assert repeat.similarity > 0.99
    assert fresh is None


def test_two_ideas_of_the_same_batch_cannot_both_pass(db: Session, niche: Niche) -> None:
    novelty = NoveltyFilter(db, niche.id, FakeEmbedder())

    first, second = novelty.check(["Même idée exacte ici", "Même idée exacte ici"])

    assert first is None
    assert second.script_id is None


def test_script_vectors_are_computed_once_and_stored(db: Session, niche: Niche) -> None:
    add_script(db, niche, [])
    embedder = FakeEmbedder()

    NoveltyFilter(db, niche.id, embedder)
    NoveltyFilter(db, niche.id, embedder)

    assert embedder.calls == 1


def test_changing_embedding_model_recomputes_vectors(
    db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_script(db, niche, [])
    embedder = FakeEmbedder()
    NoveltyFilter(db, niche.id, embedder)

    monkeypatch.setenv("EMBEDDING_MODEL", "autre-modele")
    get_settings.cache_clear()
    NoveltyFilter(db, niche.id, embedder)

    assert embedder.calls == 2


def test_other_niches_do_not_count_as_repeats(db: Session, niche: Niche) -> None:
    other = other_niche(db)
    novelty_niche_script(db, other, "Angle commun", "Croyance commune")
    novelty = NoveltyFilter(db, niche.id, FakeEmbedder())

    assert novelty.check([idea_text("Angle commun", "Croyance commune")]) == [None]


class RepeatingClaude(FakeAgentClaude):
    """Propose des idées déjà produites au premier appel, de nouvelles au second."""

    def __init__(self, repeated: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.repeated = repeated

    def _reply(self, prompt, schema, variables):
        if prompt != "script_ideas":
            return super()._reply(prompt, schema, variables)
        topics = json.loads(variables["topics"])
        first = self.count("script_ideas") == 1
        return schema.model_validate(
            {
                "ideas": [
                    {
                        "combination_index": index,
                        "topic_id": topics[index]["id"],
                        "core_idea": self.repeated if first else f"Nouvelle piste {index}",
                        "angle": "Thomas ouvre sur le même gabarit",
                        "belief_attacked": "Une croyance",
                        "why": "Parce que",
                        "potential": index + 1,
                    }
                    for index in range(variables["count"])
                ]
            }
        )


def test_generate_drops_repeats_and_asks_for_new_ideas_once(db: Session, niche: Niche) -> None:
    pattern = add_pattern(db)
    known = novelty_niche_script(db, niche, "Idée déjà tournée", "", core_idea="Idée déjà tournée")
    fake = RepeatingClaude(
        "Idée déjà tournée", drafts=[script_draft([pattern.id], angle="Nouvelle version")]
    )

    result = generate(db, niche.id, fake, n=2, develop=1, embedder=FakeEmbedder())

    assert fake.count("script_ideas") == 2
    assert len(result.scripts) == 1
    assert known.id in {item.script_id for item in result.repeats}
    second = fake.calls[1]["variables"]["past_ideas"]
    assert "Idée déjà tournée" in second


def test_script_keeps_its_core_idea_and_old_scripts_fall_back(db: Session, niche: Niche) -> None:
    pattern = add_pattern(db)
    old = novelty_niche_script(db, niche, "Angle ancien", "Croyance ancienne")
    fake = FakeAgentClaude(drafts=[script_draft([pattern.id])])

    [script] = generate(db, niche.id, fake, n=1, develop=1, seed=1).scripts

    assert load_content(script).core_idea == "Idée de fond 0"
    past = NoveltyFilter(db, niche.id, FakeEmbedder()).past
    assert past == [idea_text("Angle ancien", "Croyance ancienne"), "Idée de fond 0"]
    assert old.id in {item.id for item in db.scalars(select(Script))}


@pytest.mark.embeddings
def test_threshold_separates_rewordings_from_different_ideas_with_the_real_model() -> None:
    """Mesuré : reformulations à 0,67 ou plus, idées différentes à 0,52 ou moins."""
    from app.patterns.embeddings import embed_texts

    rewordings = [
        (
            "Garde l'école et choisis un seul service au lieu de sauter d'un modèle à l'autre.",
            "Pas besoin de lâcher l'école : concentre-toi sur un seul service.",
        ),
        (
            "Le succès en business de service vient d'un système répétable, pas de la chance.",
            "Ce n'est pas la chance qui fait réussir un service, c'est un processus répété.",
        ),
    ]
    different = (
        "Un service local rapporte plus que l'argent en ligne, même si ça paraît moins attirant.",
        "Le pire cas d'une business de service coûte quelques centaines de dollars.",
    )
    threshold = get_settings().idea_similarity_threshold

    for first, second in rewordings:
        vectors = embed_texts([first, second])
        assert float(vectors[0] @ vectors[1]) >= threshold
    vectors = embed_texts(list(different))
    assert float(vectors[0] @ vectors[1]) < threshold


def test_past_ideas_are_sent_to_claude(db: Session, niche: Niche) -> None:
    pattern = add_pattern(db)
    novelty_niche_script(db, niche, "Vieille idée", "Vieille croyance")
    fake = FakeAgentClaude(drafts=[script_draft([pattern.id])])

    generate(db, niche.id, fake, n=1, develop=1, seed=1)

    [call] = [item for item in fake.calls if item["prompt"] == "script_ideas"]
    assert call["version"] == "v2"
    assert "Vieille idée" in call["variables"]["past_ideas"]


def test_generate_gives_up_when_only_repeats_come_back(db: Session, niche: Niche) -> None:
    add_pattern(db)
    novelty_niche_script(db, niche, "Toujours pareil", "", core_idea="Toujours pareil")

    class Stubborn(RepeatingClaude):
        def count(self, prompt: str) -> int:
            return 1

    fake = Stubborn("Toujours pareil")

    with pytest.raises(GenerationError, match="répètent"):
        generate(db, niche.id, fake, n=1, develop=1, embedder=FakeEmbedder())
    assert fake.calls[-1]["prompt"] == "script_ideas"
    assert db.scalar(select(func.count()).select_from(Script)) == 1


def test_generate_can_force_exactly_one_exploration_script(db: Session, niche: Niche) -> None:
    for index in range(3):
        add_pattern(db, f"P{index}")

    result = generate(db, niche.id, PatternAwareScripts(), n=5, develop=3, seed=1, exploration=1)

    flags = [load_content(item).experiment for item in result.scripts]
    assert flags.count(True) == 1


def test_adapt_drops_a_variant_that_repeats_a_script(
    db: Session, niche: Niche, sample_video: Path
) -> None:
    from app.db.models import Analysis, Pattern, VideoPattern

    video_id = ingest(str(sample_video), deps=make_deps()).video.id
    pattern = Pattern(type="hook", name="P", description="d", template="t", examples=[])
    db.add(pattern)
    db.flush()
    db.add(VideoPattern(video_id=video_id, pattern_id=pattern.id, role="opening"))
    db.add(
        Analysis(
            video_id=video_id,
            prompt_version="analysis/v1",
            model="m",
            json_result={"analysis": sample_analysis()},
        )
    )
    db.commit()
    known = novelty_niche_script(db, niche, "x", "y", core_idea="Idée de fond 0")
    fake = FakeAgentClaude(drafts=[script_draft([pattern.id], angle="Autre")])

    def exact(texts: list[str]):
        from tests.conftest import distinct_vectors

        return distinct_vectors(texts)

    result = adapt(db, video_id, niche.id, fake, 2, embedder=exact)

    assert len(result.scripts) == 1
    assert [item.script_id for item in result.repeats] == [known.id]
    assert fake.calls[1]["version"] == "v2"


client = TestClient(app)


@pytest.fixture
def api_fake():
    fake = FakeAgentClaude()
    app.dependency_overrides[get_claude] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


def test_api_chat_and_resume(db: Session, niche: Niche, api_fake: FakeAgentClaude) -> None:
    api_fake.turns = [
        tool_turn("get_niche"),
        text_turn("Selon [script 404], rien."),
        text_turn("Encore rien."),
    ]

    first = client.post("/agent/chat", json={"niche_id": niche.id, "message": "Salut"})
    body = first.json()
    second = client.post(
        "/agent/chat",
        json={"niche_id": niche.id, "message": "Et ?", "session_id": body["session_id"]},
    )

    assert first.status_code == 200
    assert body["steps"][0] == {
        "kind": "tool_start",
        "name": "get_niche",
        "detail": "{}",
        "is_error": False,
    }
    assert body["unverified"] == ["script 404"]
    assert second.json()["session_id"] == body["session_id"]
    assert second.json()["answer"] == "Encore rien."

    session = client.get(f"/agent/sessions/{body['session_id']}").json()
    assert len(session["messages"]) == 6


def test_api_chat_unknown_niche_and_session(db: Session, niche: Niche, api_fake) -> None:
    assert client.post("/agent/chat", json={"niche_id": 999, "message": "x"}).status_code == 404
    response = client.post(
        "/agent/chat", json={"niche_id": niche.id, "message": "x", "session_id": 999}
    )
    assert response.status_code == 404
    assert client.get("/agent/sessions/999").status_code == 404


def test_api_chat_without_key_is_a_503(db: Session, niche: Niche, api_fake) -> None:
    api_fake.turns = [ClaudeCredentialsError("Aucune clé")]

    response = client.post("/agent/chat", json={"niche_id": niche.id, "message": "x"})

    assert response.status_code == 503


def test_api_journal_budget_notifications_and_reports(db: Session, niche: Niche, api_fake) -> None:
    api_fake.turns = [text_turn("Ok")]
    client.post("/agent/chat", json={"niche_id": niche.id, "message": "x"})
    spend(db, 123)
    row = Notification(kind="stats_reminder", title="Stats à 24 h")
    db.add(row)
    db.add(WeeklyReport(niche_id=niche.id, week_start="2026-11-02", markdown="# Rapport"))
    db.commit()

    assert client.get("/agent/actions").json()[0]["action"] == "chat_turn"
    assert client.get("/agent/budget").json()["used"] == 123
    [notification] = client.get("/agent/notifications").json()
    assert notification["title"] == "Stats à 24 h"
    assert client.post(f"/agent/notifications/{row.id}/read").json()["read_at"]
    assert client.get("/agent/notifications").json() == []
    assert len(client.get("/agent/notifications?unread=false").json()) == 1
    assert client.post("/agent/notifications/999/read").status_code == 404
    assert client.get(f"/agent/reports?niche_id={niche.id}").json()[0]["markdown"] == "# Rapport"


def test_patch_script_sets_publication_date(db: Session, niche: Niche) -> None:
    script = add_script(db, niche, [])

    response = client.patch(
        f"/scripts/{script.id}",
        json={"status": "published", "published_at": "2026-11-01T12:00:00Z"},
    )

    assert response.status_code == 200
    assert response.json()["published_at"].startswith("2026-11-01T12:00:00")


runner = CliRunner()


def test_cli_chat_shows_steps_and_answer(
    db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeAgentClaude([tool_turn("get_niche", say="Je regarde."), text_turn("Ta niche.")])
    monkeypatch.setattr(cli_module, "default_claude", lambda: fake)

    result = runner.invoke(cli, ["chat", "--niche", str(niche.id)], input="Qui suis-je ?\n/fin\n")

    assert result.exit_code == 0, result.output
    assert "Je regarde." in result.output
    assert "→ get_niche" in result.output
    assert "Ta niche." in result.output


def test_cli_chat_can_resume(db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAgentClaude([text_turn("Un"), text_turn("Deux")])
    monkeypatch.setattr(cli_module, "default_claude", lambda: fake)
    agent_session = start_session(db, niche.id)
    chat(db, agent_session, "x", fake)

    args = ["chat", "--niche", str(niche.id), "--resume", str(agent_session.id)]
    result = runner.invoke(cli, args, input="Encore\n/fin\n")

    assert result.exit_code == 0, result.output
    assert "Deux" in result.output


def test_cli_chat_unknown_session(
    db: Session, niche: Niche, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "default_claude", FakeAgentClaude)

    result = runner.invoke(cli, ["chat", "--niche", str(niche.id), "--resume", "999"], input="")

    assert result.exit_code == 1


def test_cli_publish_starts_the_reminders(db: Session, niche: Niche) -> None:
    script = add_script(db, niche, [])

    result = runner.invoke(cli, ["publish", str(script.id), "--at", "2026-11-01T12:00:00"])

    assert result.exit_code == 0, result.output
    db.expire_all()
    assert script.status is ScriptStatus.PUBLISHED
    assert script.published_at is not None


def test_cli_publish_unknown(db: Session) -> None:
    assert runner.invoke(cli, ["publish", "999"]).exit_code == 1


def test_cli_notifications_and_actions(db: Session, niche: Niche) -> None:
    db.add(Notification(kind="budget", title="Budget atteint", body="Détail"))
    db.commit()
    run_tool(context(db, niche), "learning_report", {})

    first = runner.invoke(cli, ["notifications"])
    second = runner.invoke(cli, ["notifications"])
    actions = runner.invoke(cli, ["actions"])

    assert "Budget atteint" in first.output
    assert "Aucune notification" in second.output
    assert "learning_report" in actions.output
    assert "Budget du jour" in actions.output


def test_saving_stats_dates_the_publication_from_the_horizon(db: Session, niche: Niche) -> None:
    script = add_script(db, niche, [])
    measured = datetime.now(UTC) - timedelta(hours=1)
    data = stats()
    data.measured_at = measured

    save_performance(db, script, data)

    assert script.published_at == measured - timedelta(hours=72)


def test_prompts_load() -> None:
    assert "{{" not in load_prompt("agent_chat", "v1").system
    assert load_prompt("agent_chat", "v1").render_user({"memory": "x"})
    assert load_prompt("weekly_report", "v1")
