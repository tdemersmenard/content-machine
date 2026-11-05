"""Adaptation, génération d'idées, critique et garde-fous."""

import json
from pathlib import Path

import anthropic
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app import cli as cli_module
from app.cli import cli
from app.db.models import (
    Analysis,
    Lesson,
    LessonScope,
    Pattern,
    Script,
    ScriptStatus,
    Transcript,
    VideoPattern,
)
from app.generate.guards import (
    ScriptGuard,
    copied_runs,
    flag_numbers,
    ngrams,
    normalize_number,
    numbers_in,
    verification_notes,
)
from app.generate.report import render_script
from app.generate.schemas import ScriptDraft
from app.generate.service import (
    GenerationError,
    _polish,
    _predict,
    adapt,
    avoided_pattern_ids,
    generate,
    load_content,
    script_path,
    thompson_combination,
    update_script,
)
from app.ingest.jobs import ingest
from app.llm.client import get_claude
from app.llm.errors import ClaudeInvalidOutputError
from app.llm.prompts import load_prompt
from app.main import app
from app.niche.profile import load_profile, set_profile
from tests.analysis_helpers import sample_analysis
from tests.ingest_helpers import make_deps
from tests.script_helpers import FakeScriptClaude, script_draft
from tests.test_niche import make_niche


@pytest.fixture
def niche(db: Session):
    return make_niche(db)


@pytest.fixture
def source(db: Session, sample_video: Path) -> tuple[int, list[int]]:
    """Une vidéo analysée, liée à 2 patterns."""
    video_id = ingest(str(sample_video), deps=make_deps()).video.id
    db.add(
        Analysis(
            video_id=video_id,
            prompt_version="analysis/v1",
            model="m",
            json_result={"analysis": sample_analysis()},
        )
    )
    patterns = [
        Pattern(type="hook", name="P1", description="d", template="[x]", examples=[]),
        Pattern(type="ending", name="P2", description="d", template="[x]", examples=[]),
    ]
    db.add_all(patterns)
    db.flush()
    db.add_all([VideoPattern(video_id=video_id, pattern_id=p.id, role="opening") for p in patterns])
    db.commit()
    return video_id, [item.id for item in patterns]


def drafts_for(ids: list[int], count: int = 3) -> list[dict]:
    return [script_draft(ids) for _ in range(count)]


def test_adapt_runs_skeleton_transposition_writing_and_critic(db: Session, niche, source) -> None:
    video_id, ids = source
    fake = FakeScriptClaude(drafts=drafts_for(ids))

    result = adapt(db, video_id, niche.id, fake)

    assert len(result.scripts) == 3
    assert result.failures == []
    assert [c["prompt"] for c in fake.calls][:4] == [
        "script_skeleton",
        "script_transpose",
        "script_write",
        "script_critic",
    ]
    script = result.scripts[0]
    assert script.source_video_id == video_id
    assert script.niche_id == niche.id
    assert script.pattern_ids == ids
    assert script.prompt_version == "script_write/v1"
    assert script.status is ScriptStatus.DRAFT


def test_writer_receives_skeleton_variant_profile_and_patterns(db: Session, niche, source) -> None:
    video_id, ids = source
    fake = FakeScriptClaude(drafts=drafts_for(ids))

    adapt(db, video_id, niche.id, fake)

    brief = json.loads(
        next(c for c in fake.calls if c["prompt"] == "script_write")["variables"]["brief"]
    )
    assert brief["skeleton"]["beats"][0]["mechanism"] == "Installe une tension abstraite"
    assert brief["variant"]["angle"] == "Angle 0"
    assert brief["profile"]["voice"]["tone"] == "Direct"
    assert {item["name"] for item in brief["patterns"]} == {"P1", "P2"}


def test_adapt_requires_an_analysis(db: Session, niche, sample_video: Path) -> None:
    video_id = ingest(str(sample_video), deps=make_deps()).video.id
    fake = FakeScriptClaude()

    with pytest.raises(GenerationError, match="cm analyze"):
        adapt(db, video_id, niche.id, fake)

    assert fake.calls == []


def test_adapt_unknown_targets(db: Session, niche) -> None:
    with pytest.raises(LookupError):
        adapt(db, 999, niche.id, FakeScriptClaude())
    with pytest.raises(LookupError):
        adapt(db, 1, 999, FakeScriptClaude())


def test_skeleton_schema_refuses_words_of_the_source(db: Session, niche, source) -> None:
    video_id, ids = source
    fake = FakeScriptClaude(drafts=drafts_for(ids))
    adapt(db, video_id, niche.id, fake)
    schema = fake.calls[0]["schema"]

    with pytest.raises(ValidationError, match="repris de la vidéo"):
        schema.model_validate(
            {
                "beats": [
                    {
                        "role": "setup",
                        "mechanism": "Dire how to double your views ici",
                        "start_pct": 0,
                        "end_pct": 10,
                    }
                ]
            }
        )


def test_transposition_needs_one_equivalent_per_beat_and_real_topics(
    db: Session, niche, source
) -> None:
    video_id, ids = source
    fake = FakeScriptClaude(drafts=drafts_for(ids, 1))
    adapt(db, video_id, niche.id, fake, n_variantes=1)
    schema = fake.calls[1]["schema"]
    beat = {"pain": "d", "belief": "", "proof": "p", "example": ""}
    variant = {"angle": "a", "belief_attacked": "b", "topic_ids": []}

    with pytest.raises(ValidationError, match="équivalent par beat"):
        schema.model_validate({"variants": [{**variant, "beats": [{"beat_index": 0, **beat}]}]})
    with pytest.raises(ValidationError, match="Sujets inconnus"):
        schema.model_validate(
            {
                "variants": [
                    {
                        **variant,
                        "topic_ids": ["faux"],
                        "beats": [{"beat_index": i, **beat} for i in range(2)],
                    }
                ]
            }
        )
    with pytest.raises(ValidationError, match="exactement 1"):
        schema.model_validate({"variants": []})


def test_ngrams_and_copied_runs() -> None:
    source = ngrams("Tu sais ce qui est bien avec le fait", 6)

    assert copied_runs("Bon, tu sais ce qui est bien, mon gars", source, 6) == [
        "tu sais ce qui est bien"
    ]
    assert copied_runs("Tu sais ce qui est pas bien", source, 6) == []


def guard(**kwargs) -> ScriptGuard:
    return ScriptGuard(allowed_pattern_ids=frozenset({1}), **kwargs)


def draft(**overrides) -> ScriptDraft:
    return ScriptDraft.model_validate(script_draft([1], **overrides))


def test_five_copied_words_pass_and_six_are_refused() -> None:
    source = ngrams("alpha beta gamma delta epsilon zeta eta", 6)

    five = draft(ending="alpha beta gamma delta epsilon pis autre chose")
    six = draft(ending="alpha beta gamma delta epsilon zeta")

    assert guard(source_runs=source).violations(five) == []
    assert "repris de la source" in guard(source_runs=source).violations(six)[0]


def test_copy_check_covers_hooks_screen_text_and_caption() -> None:
    source = ngrams("alpha beta gamma delta epsilon zeta", 6)
    text = "alpha beta gamma delta epsilon zeta"
    data = script_draft([1])
    data["hooks"][1]["on_screen_text"] = text

    assert guard(source_runs=source).violations(ScriptDraft.model_validate(data))
    assert guard(source_runs=source).violations(draft(caption=text))


def test_never_say_is_refused() -> None:
    violations = guard(never_say=("que c'est facile",)).violations(
        draft(ending="Tu vas voir que c'est facile")
    )

    assert "Formule interdite" in violations[0]


def test_structure_violations() -> None:
    data = script_draft([1, 99])
    data["hooks"] = data["hooks"][:2]
    data["open_loops"][0]["closed_beat"] = 8

    errors = " ".join(guard().violations(ScriptDraft.model_validate(data)))

    assert "3 hooks" in errors
    assert "pattern_ids inconnus" in errors
    assert "Boucle hors des beats" in errors


def test_guarded_schema_rejects_so_the_wrapper_can_retry() -> None:
    schema = guard(never_say=("mot banni",)).schema()

    with pytest.raises(ValidationError, match="Formule interdite"):
        schema.model_validate(script_draft([1], ending="Le mot banni"))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("800", "800"), ("100 000", "100000"), ("100k", "100000"), ("1,5", "1.5"), ("2,000", "2000")],
)
def test_normalize_number(raw: str, expected: str) -> None:
    assert normalize_number(raw) == expected


def test_numbers_in_finds_every_form() -> None:
    assert numbers_in("près de 100k$, 18 ans et 2 000 balles") == {"100000", "18", "2000"}


def test_unknown_number_is_flagged_and_known_one_is_kept() -> None:
    text, flagged = flag_numbers("J'ai fait 100k$ pis j'ai 4 chars", {"100000"})

    assert text == "J'ai fait 100k$ pis j'ai [À VÉRIFIER : 4] chars"
    assert flagged == ["4"]


def test_small_counts_are_not_claims() -> None:
    text, flagged = flag_numbers("Trouve 1 client, évite les 2 erreurs, en 3 étapes", set())

    assert text == "Trouve 1 client, évite les 2 erreurs, en 3 étapes"
    assert flagged == []


@pytest.mark.parametrize("raw", ["3 %", "2$", "3 k", "4 clients", "1,5 %"])
def test_small_numbers_with_a_unit_or_above_the_limit_stay_flagged(raw: str) -> None:
    assert flag_numbers(f"Tu fais {raw} net", set())[1]


def test_verification_notes_collect_numbers_and_notes_from_claude() -> None:
    notes = verification_notes(
        [
            "J'ai [À VÉRIFIER : 4] chars",
            "[À VÉRIFIER : confirmer les études] pis [À VÉRIFIER]",
            "J'ai [À VÉRIFIER : 4] chars encore",
        ]
    )

    assert notes == ["4", "confirmer les études", "non précisé"]


def test_note_written_by_claude_reaches_the_top_of_the_script(db: Session, niche, source) -> None:
    video_id, ids = source
    data = script_draft(ids, ending="J'étais à l'école [À VÉRIFIER : confirmer les études]")
    fake = FakeScriptClaude(drafts=[data])

    (script,) = adapt(db, video_id, niche.id, fake, n_variantes=1).scripts

    assert load_content(script).to_verify == ["confirmer les études"]


def test_text_already_in_brackets_is_left_alone() -> None:
    text, flagged = flag_numbers("[À VÉRIFIER : 12 clients] et [À VÉRIFIER]", set())

    assert text == "[À VÉRIFIER : 12 clients] et [À VÉRIFIER]"
    assert flagged == []


def test_invented_numbers_are_flagged_in_the_stored_script(db: Session, niche, source) -> None:
    video_id, ids = source
    data = script_draft(ids, ending="En 7 jours tu fais 45 000 piastres")
    data["hooks"][0]["spoken"] = "J'ai perdu 800 piastres pis 9999 clients"
    fake = FakeScriptClaude(drafts=[data])

    (script,) = adapt(db, video_id, niche.id, fake, n_variantes=1).scripts

    content = load_content(script)
    assert "[À VÉRIFIER : 7]" in content.ending
    assert "[À VÉRIFIER : 45 000]" in content.ending
    assert "[À VÉRIFIER : 9999]" in content.hooks[0].spoken
    assert set(content.to_verify) == {"7", "45 000", "9999", "800"}


def test_creator_facts_reach_the_writer_and_trust_their_numbers(db: Session, niche, source) -> None:
    video_id, ids = source
    profile = load_profile(niche)
    profile.creator_facts = ["Avait 17 ans au début"]
    set_profile(niche, profile)
    db.commit()
    fake = FakeScriptClaude(drafts=[script_draft(ids, ending="À 17 ans j'ai parti ça")])

    (script,) = adapt(db, video_id, niche.id, fake, n_variantes=1).scripts

    brief = json.loads(
        next(c for c in fake.calls if c["prompt"] == "script_write")["variables"]["brief"]
    )
    assert brief["profile"]["creator_facts"] == ["Avait 17 ans au début"]
    assert load_content(script).to_verify == []


def test_number_from_the_profile_is_trusted(db: Session, niche, source) -> None:
    video_id, ids = source
    profile = load_profile(niche)
    profile.avatar.already_tried = ["Dropshipping, a perdu 800 piastres"]
    set_profile(niche, profile)
    db.commit()
    fake = FakeScriptClaude(drafts=[script_draft(ids, ending="Il a perdu 800 piastres")])

    (script,) = adapt(db, video_id, niche.id, fake, n_variantes=1).scripts

    assert load_content(script).ending == "Il a perdu 800 piastres"
    assert load_content(script).to_verify == []


def test_duration_comes_from_the_source_speed_without_a_voice_profile(
    db: Session, niche, source
) -> None:
    video_id, ids = source
    fake = FakeScriptClaude(drafts=[script_draft(ids)])

    (script,) = adapt(db, video_id, niche.id, fake, n_variantes=1).scripts

    content = load_content(script)
    assert content.duration_source == "source_video"
    assert content.words_per_second == pytest.approx(10 / 9, abs=0.05)
    words = 7 + 6 + 3  # Hook retenu, puis les deux beats.
    assert content.estimated_duration_s == pytest.approx(words / content.words_per_second, abs=0.2)
    assert content.beats[0].target_start_s > 0
    assert content.beats[1].target_start_s == content.beats[0].target_end_s


def test_duration_uses_my_own_speed_when_it_is_known(db: Session, niche, source) -> None:
    from app.niche.schemas import VoiceProfile

    video_id, ids = source
    profile = load_profile(niche)
    profile.voice_profile = VoiceProfile(
        video_ids=[1],
        avg_sentence_words=8,
        median_sentence_words=7,
        short_sentence_ratio=0.2,
        words_per_second=3.0,
        silences_per_minute=5,
        verbal_tics=[],
        signature_expressions=[],
        style_summary="s",
        prompt_version="voice_extract/v1",
    )
    set_profile(niche, profile)
    db.commit()
    fake = FakeScriptClaude(drafts=[script_draft(ids)])

    (script,) = adapt(db, video_id, niche.id, fake, n_variantes=1).scripts

    content = load_content(script)
    assert content.duration_source == "voice_profile"
    assert content.estimated_duration_s == pytest.approx(16 / 3, abs=0.05)


def run_polish(scores: list[int], drafts: list[dict] | None = None):
    from app.generate.service import Brief

    fake = FakeScriptClaude(scores=scores, drafts=drafts or drafts_for([1], 5))
    profile = _profile()
    brief = Brief(
        mode="adapt",
        niche=None,
        profile=profile,
        pattern_ids=[1],
        topic_ids=[],
        content={},
        guard=guard(),
        words_per_second=3,
        duration_source="default",
    )
    return _polish(fake, brief, draft()), fake


def _profile():
    from app.niche.schemas import NicheProfile
    from tests.niche_helpers import draft as niche_draft

    return NicheProfile.model_validate(niche_draft())


def test_good_hook_is_not_rewritten() -> None:
    (_, final, history), fake = run_polish([8])

    assert fake.count("script_rewrite") == 0
    assert final.hook_score == 8
    assert len(history) == 1


def test_weak_hook_is_rewritten_until_it_reaches_8() -> None:
    (_, final, history), fake = run_polish([5, 8])

    assert fake.count("script_rewrite") == 1
    assert final.hook_score == 8
    assert [item.hook_score for item in history] == [5, 8]


def test_rewrites_stop_after_two_iterations() -> None:
    (_, final, _), fake = run_polish([3, 4, 5, 9])

    assert fake.count("script_rewrite") == 2
    assert fake.count("script_critic") == 3
    assert final.hook_score == 5


def test_a_worse_rewrite_is_thrown_away() -> None:
    better = script_draft([1], internal_title="Meilleure")
    worse = script_draft([1], internal_title="Pire")

    (best, final, history), _ = run_polish([5, 7, 2], [better, worse])

    assert best.internal_title == "Meilleure"
    assert final.hook_score == 7
    assert len(history) == 3


def test_failed_rewrite_keeps_the_previous_version() -> None:
    from app.generate.service import Brief

    class Failing(FakeScriptClaude):
        def _reply(self, prompt, schema, variables):
            if prompt == "script_rewrite":
                raise ClaudeInvalidOutputError("invalide", "x")
            return super()._reply(prompt, schema, variables)

    brief = Brief(
        mode="adapt",
        niche=None,
        profile=_profile(),
        pattern_ids=[1],
        topic_ids=[],
        content={},
        guard=guard(),
        words_per_second=3,
        duration_source="default",
    )

    best, final, _ = _polish(Failing(scores=[5]), brief, draft())

    assert best.internal_title == "Le vrai problème"
    assert final.hook_score == 5


def test_swipe_points_must_quote_the_script() -> None:
    (_, final, _), _ = run_polish([9])

    assert [item.quote for item in final.swipe_points] == ["Un deux trois"]


def test_critic_plays_the_avatar() -> None:
    _, fake = run_polish([9])

    assert "Salarié de 28 ans" in fake.calls[0]["variables"]["avatar"]


def test_predicted_score_formula() -> None:
    patterns = [Pattern(score_alpha=3.0, score_beta=1.0), Pattern(score_alpha=1.0, score_beta=1.0)]

    predicted = _predict(8, patterns)

    assert predicted.value == pytest.approx(48 + 25)
    assert predicted.pattern_mean == 0.625
    assert _predict(5, []).value == 50


def test_predicted_score_is_stored_for_later_comparison(db: Session, niche, source) -> None:
    video_id, ids = source
    fake = FakeScriptClaude(drafts=[script_draft(ids)], scores=[9])

    (script,) = adapt(db, video_id, niche.id, fake, n_variantes=1).scripts

    assert script.predicted_score == pytest.approx(54 + 20)


def test_one_failed_variant_keeps_the_others(db: Session, niche, source) -> None:
    video_id, ids = source

    class FailsOnce(FakeScriptClaude):
        def _reply(self, prompt, schema, variables):
            if prompt == "script_write" and self.count("script_write") == 2:
                raise ClaudeInvalidOutputError("Guard", "x")
            return super()._reply(prompt, schema, variables)

    result = adapt(db, video_id, niche.id, FailsOnce(drafts=drafts_for(ids, 2)))

    assert len(result.scripts) == 2
    assert result.failures == ["Guard"]


def test_all_variants_failing_raises(db: Session, niche, source) -> None:
    video_id, _ = source

    class AlwaysFails(FakeScriptClaude):
        def _reply(self, prompt, schema, variables):
            if prompt == "script_write":
                raise ClaudeInvalidOutputError("Guard", "x")
            return super()._reply(prompt, schema, variables)

    with pytest.raises(GenerationError, match="Aucun script"):
        adapt(db, video_id, niche.id, AlwaysFails(), n_variantes=1)


def library(db: Session) -> list[Pattern]:
    patterns = [
        Pattern(type="hook", name="H1", description="d", template="[x]", examples=[]),
        Pattern(type="hook", name="H2", description="d", template="[x]", examples=[]),
        Pattern(type="ending", name="E1", description="d", template="[x]", examples=[]),
    ]
    db.add_all(patterns)
    db.commit()
    return patterns


def test_thompson_takes_one_pattern_per_type() -> None:
    patterns = [
        Pattern(type="hook", name="a", score_alpha=1.0, score_beta=1.0),
        Pattern(type="hook", name="b", score_alpha=1.0, score_beta=1.0),
        Pattern(type="ending", name="c", score_alpha=1.0, score_beta=1.0),
    ]

    chosen = thompson_combination(patterns, np.random.default_rng(0))

    assert sorted(item.type for item in chosen) == ["ending", "hook"]


def test_thompson_favors_winners_but_still_explores() -> None:
    winner = Pattern(type="hook", name="winner", score_alpha=30.0, score_beta=10.0)
    new = Pattern(type="hook", name="new", score_alpha=1.0, score_beta=1.0)
    rng = np.random.default_rng(1)

    picks = [thompson_combination([winner, new], rng)[0].name for _ in range(1000)]

    assert 0.6 < picks.count("winner") / 1000 < 0.9
    assert picks.count("new") > 100


def test_thompson_rarely_picks_a_proven_loser() -> None:
    loser = Pattern(type="hook", name="loser", score_alpha=2.0, score_beta=40.0)
    new = Pattern(type="hook", name="new", score_alpha=1.0, score_beta=1.0)
    rng = np.random.default_rng(2)

    picks = [thompson_combination([loser, new], rng)[0].name for _ in range(1000)]

    assert picks.count("loser") < 100


def test_generate_proposes_n_ideas_and_develops_the_best(db: Session, niche) -> None:
    patterns = library(db)
    ids = [item.id for item in patterns]

    class AnyCombination(FakeScriptClaude):
        def _reply(self, prompt, schema, variables):
            if prompt == "script_write":
                brief = json.loads(variables["brief"])
                chosen = [item["id"] for item in brief["patterns"]]
                return schema.model_validate(script_draft(chosen))
            return super()._reply(prompt, schema, variables)

    fake = AnyCombination()

    result = generate(db, niche.id, fake, "Plus d'abonnés", n=5, develop=2, seed=1)

    assert len(result.scripts) == 2
    assert fake.count("script_ideas") == 1
    assert fake.count("script_write") == 2
    written = [
        json.loads(c["variables"]["brief"]) for c in fake.calls if c["prompt"] == "script_write"
    ]
    assert [item["idea"]["potential"] for item in written] == [5, 4]
    assert written[0]["objective"] == "Plus d'abonnés"
    assert all(set(script.pattern_ids) <= set(ids) for script in result.scripts)
    assert load_content(result.scripts[0]).mode == "generate"
    assert load_content(result.scripts[0]).duration_source == "default"
    assert load_content(result.scripts[0]).topic_ids


def test_generate_defaults_to_the_goal_of_the_profile(db: Session, niche) -> None:
    library(db)
    fake = FakeScriptClaude()

    with pytest.raises(GenerationError):
        generate(db, niche.id, fake, n=1, develop=1, seed=1)

    assert fake.calls[0]["variables"]["objective"] == "10 appels par mois"


def test_lessons_exclude_patterns_and_reach_the_prompts(db: Session, niche) -> None:
    patterns = library(db)
    bad = patterns[0]
    db.add_all(
        [
            Lesson(
                scope=LessonScope.GLOBAL,
                statement="Ce hook ne marche plus",
                evidence=[{"pattern_id": bad.id, "effect": "avoid"}],
            ),
            Lesson(
                scope=LessonScope.GLOBAL,
                statement="Leçon désactivée",
                active=False,
                evidence=[{"pattern_id": patterns[1].id, "effect": "avoid"}],
            ),
        ]
    )
    db.commit()
    fake = FakeScriptClaude()

    with pytest.raises(GenerationError):
        generate(db, niche.id, fake, n=3, develop=1, seed=1)

    call = fake.calls[0]["variables"]
    combinations = json.loads(call["combinations"])
    names = {item["name"] for combo in combinations for item in combo}
    assert bad.name not in names
    assert "H2" in names
    assert "Ce hook ne marche plus" in call["lessons"]
    assert "Leçon désactivée" not in call["lessons"]


def test_avoided_ids_ignore_other_evidence() -> None:
    lessons = [
        Lesson(evidence=[{"pattern_id": 1, "effect": "boost"}, "texte", {"effect": "avoid"}])
    ]

    assert avoided_pattern_ids(lessons) == set()


def test_generate_needs_patterns_and_topics(db: Session, niche) -> None:
    with pytest.raises(GenerationError, match="patterns"):
        generate(db, niche.id, FakeScriptClaude())

    library(db)
    niche.topics = []
    db.commit()
    with pytest.raises(GenerationError, match="sujets"):
        generate(db, niche.id, FakeScriptClaude())


def test_ideas_schema_refuses_unknown_topics(db: Session, niche) -> None:
    from app.generate.service import _ideas_schema

    schema = _ideas_schema(1, 1, {"t1"})
    idea = {"angle": "a", "belief_attacked": "b", "why": "w", "potential": 5}

    with pytest.raises(ValidationError, match="Sujet inconnu"):
        schema.model_validate({"ideas": [{"combination_index": 0, "topic_id": "x", **idea}]})
    with pytest.raises(ValidationError, match="Combinaison inconnue"):
        schema.model_validate({"ideas": [{"combination_index": 3, "topic_id": "t1", **idea}]})
    with pytest.raises(ValidationError, match="exactement 1"):
        schema.model_validate({"ideas": []})


def test_generate_also_refuses_copying_pattern_examples(db: Session, niche) -> None:
    db.add(
        Pattern(
            type="hook",
            name="H",
            description="d",
            template="[x]",
            examples=[
                {"video_id": 1, "excerpt": "alpha beta gamma delta epsilon zeta", "timestamp": 0}
            ],
        )
    )
    db.commit()

    class Copying(FakeScriptClaude):
        def _reply(self, prompt, schema, variables):
            if prompt == "script_write":
                chosen = [item["id"] for item in json.loads(variables["brief"])["patterns"]]
                schema.model_validate(
                    script_draft(chosen, ending="alpha beta gamma delta epsilon zeta")
                )
            return super()._reply(prompt, schema, variables)

    with pytest.raises(GenerationError, match="Aucun script") as caught:
        try:
            generate(db, niche.id, Copying(), n=1, develop=1, seed=1)
        except ValidationError as exc:
            raise GenerationError(f"Aucun script : {exc}") from exc

    assert "repris de la source" in str(caught.value)


def test_markdown_has_every_section(db: Session, niche, source) -> None:
    video_id, ids = source
    fake = FakeScriptClaude(drafts=[script_draft(ids, ending="En 7 jours")], scores=[9])

    (script,) = adapt(db, video_id, niche.id, fake, n_variantes=1).scripts

    report = script_path(script.id).read_text("utf-8")
    for part in (
        "# Le vrai problème",
        "**Angle :**",
        "**Croyance attaquée :**",
        "**À vérifier avant de tourner :** 7",
        "## Hooks",
        "**Hook 3 (Contrarien)**",
        "### Beat 0 : Setup",
        "- Jeu : Pause",
        "## Boucles ouvertes",
        "**CTA :** Abonne-toi",
        "#business",
        "## Pourquoi ces choix",
        "le débit de la vidéo source",
        "Note du hook : 9/10",
    ):
        assert part in report


def make_script(db: Session, niche, source) -> Script:
    video_id, ids = source
    fake = FakeScriptClaude(drafts=[script_draft(ids)])
    return adapt(db, video_id, niche.id, fake, n_variantes=1).scripts[0]


def test_edit_recomputes_timing_and_rewrites_the_file(db: Session, niche, source) -> None:
    script = make_script(db, niche, source)
    before = load_content(script).estimated_duration_s

    update_script(
        db, script, changes={"ending": "Fin modifiée", "hooks": load_hooks(script, "Très court")}
    )

    content = load_content(script)
    assert content.ending == "Fin modifiée"
    assert content.estimated_duration_s < before
    assert "Fin modifiée" in script_path(script.id).read_text("utf-8")


def load_hooks(script: Script, spoken: str) -> list[dict]:
    hooks = [dict(item) for item in script.content["hooks"]]
    hooks[0]["spoken"] = spoken
    return hooks


def test_invalid_edit_is_refused(db: Session, niche, source) -> None:
    script = make_script(db, niche, source)

    with pytest.raises(ValidationError):
        update_script(db, script, changes={"hooks": "pas une liste"})


def test_approval_marks_topics_and_counts_patterns_once(db: Session, niche, source) -> None:
    script = make_script(db, niche, source)
    topic = niche.topics[0]["id"]
    script.content = {**script.content, "topic_ids": [topic, "disparu"]}
    db.commit()

    update_script(db, script, status=ScriptStatus.APPROVED)
    update_script(db, script, status=ScriptStatus.PUBLISHED)

    db.refresh(niche)
    assert next(t for t in niche.topics if t["id"] == topic)["status"] == "used"
    pattern = db.get(Pattern, script.pattern_ids[0])
    assert pattern.times_used == 1
    assert script.status is ScriptStatus.PUBLISHED


def test_published_video_link(db: Session, niche, source) -> None:
    script = make_script(db, niche, source)

    update_script(db, script, published_video_id=source[0])

    assert script.published_video_id == source[0]
    with pytest.raises(LookupError):
        update_script(db, script, published_video_id=999)


client = TestClient(app)


@pytest.fixture
def api_claude(source):
    fake = FakeScriptClaude(drafts=drafts_for(source[1], 5))
    app.dependency_overrides[get_claude] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


def test_adapt_route_then_list_and_patch(db: Session, niche, source, api_claude) -> None:
    created = client.post(
        "/scripts/adapt", json={"video_id": source[0], "niche_id": niche.id, "n_variantes": 2}
    )

    assert created.status_code == 201
    body = created.json()
    assert len(body["scripts"]) == 2
    assert body["scripts"][0]["markdown"].startswith("# Le vrai problème")
    assert len(client.get(f"/scripts?niche_id={niche.id}").json()) == 2

    script_id = body["scripts"][0]["id"]
    patched = client.patch(
        f"/scripts/{script_id}", json={"status": "approved", "content": {"cta": "Écris-moi"}}
    ).json()
    assert patched["status"] == "approved"
    assert patched["content"]["cta"] == "Écris-moi"
    assert [s["id"] for s in client.get("/scripts?status=approved").json()] == [script_id]


def test_generate_route_errors(db: Session, niche, api_claude) -> None:
    assert client.post("/scripts/generate", json={"niche_id": 999}).status_code == 404
    db.execute(VideoPattern.__table__.delete())
    db.execute(Pattern.__table__.delete())
    db.commit()
    assert client.post("/scripts/generate", json={"niche_id": niche.id}).status_code == 409


def test_adapt_route_errors(db: Session, niche, source, api_claude) -> None:
    db.execute(Analysis.__table__.delete())
    db.commit()

    body = {"video_id": source[0], "niche_id": niche.id}
    assert client.post("/scripts/adapt", json=body).status_code == 409
    assert client.post("/scripts/adapt", json=body | {"video_id": 999}).status_code == 404
    assert client.post("/scripts/adapt", json=body | {"n_variantes": 9}).status_code == 422


def test_patch_errors(db: Session, niche, source) -> None:
    script = make_script(db, niche, source)

    assert client.patch("/scripts/999", json={}).status_code == 404
    assert client.patch(f"/scripts/{script.id}", json={"status": "faux"}).status_code == 422
    bad = client.patch(f"/scripts/{script.id}", json={"content": {"hooks": "x"}})
    assert bad.status_code == 422
    assert (
        client.patch(f"/scripts/{script.id}", json={"published_video_id": 999}).status_code == 404
    )


def test_api_error_from_claude_is_a_502(db: Session, niche, source) -> None:
    class Down(FakeScriptClaude):
        def call_json(self, **_):
            raise anthropic.APIConnectionError(request=None, message="réseau coupé")

    app.dependency_overrides[get_claude] = lambda: Down()
    try:
        response = client.post("/scripts/adapt", json={"video_id": source[0], "niche_id": niche.id})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502


runner = CliRunner()


def test_cli_adapt(db: Session, niche, source, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeScriptClaude(drafts=drafts_for(source[1], 2))
    monkeypatch.setattr(cli_module, "default_claude", lambda: fake)

    result = runner.invoke(cli, ["adapt", str(source[0]), "--niche", str(niche.id), "-n", "2"])

    assert result.exit_code == 0, result.output
    assert "Le vrai problème" in result.output
    assert "script_1.md" in result.output.replace("\n", "")
    assert "hook 9/10" in result.output


def test_cli_ideas(db: Session, niche, monkeypatch: pytest.MonkeyPatch) -> None:
    library(db)

    class AnyCombination(FakeScriptClaude):
        def _reply(self, prompt, schema, variables):
            if prompt == "script_write":
                chosen = [item["id"] for item in json.loads(variables["brief"])["patterns"]]
                return schema.model_validate(script_draft(chosen))
            return super()._reply(prompt, schema, variables)

    fake = AnyCombination()
    monkeypatch.setattr(cli_module, "default_claude", lambda: fake)

    result = runner.invoke(cli, ["ideas", "--niche", str(niche.id), "-n", "10"])

    assert result.exit_code == 0, result.output
    assert fake.count("script_write") == 3
    assert fake.calls[0]["variables"]["count"] == 10


def test_cli_errors(db: Session, niche, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "default_claude", FakeScriptClaude)

    assert runner.invoke(cli, ["adapt", "999", "--niche", str(niche.id)]).exit_code == 1
    assert runner.invoke(cli, ["ideas", "--niche", "999"]).exit_code == 1


@pytest.mark.parametrize(
    ("name", "variables"),
    [
        ("script_skeleton", {"analysis": "{}", "patterns": "[]"}),
        (
            "script_transpose",
            {"count": 3, "skeleton": "{}", "profile": "{}", "topics": "[]", "lessons": "[]"},
        ),
        ("script_write", {"brief": "{}"}),
        ("script_rewrite", {"brief": "{}", "script": "{}", "critique": "{}"}),
        ("script_critic", {"avatar": "{}", "script": "{}"}),
        (
            "script_ideas",
            {
                "count": 5,
                "objective": "x",
                "profile": "{}",
                "topics": "[]",
                "combinations": "[]",
                "lessons": "[]",
            },
        ),
    ],
)
def test_prompts_render(name: str, variables: dict) -> None:
    assert "{{" not in load_prompt(name, "v1").render_user(variables)


def test_render_script_round_trips_from_storage(db: Session, niche, source) -> None:
    script = make_script(db, niche, source)
    stored = db.scalar(select(Script).where(Script.id == script.id))

    assert render_script(load_content(stored)).startswith("# Le vrai problème")


def test_transcript_is_read_for_the_copy_guard(db: Session, niche, source) -> None:
    video_id, ids = source
    transcript = db.get(Transcript, video_id)
    transcript.full_text = "alpha beta gamma delta epsilon zeta eta"
    db.commit()

    class Copying(FakeScriptClaude):
        def _reply(self, prompt, schema, variables):
            if prompt == "script_write":
                with pytest.raises(ValidationError, match="repris de la source"):
                    schema.model_validate(
                        script_draft(ids, ending="alpha beta gamma delta epsilon zeta")
                    )
            return super()._reply(prompt, schema, variables)

    adapt(db, video_id, niche.id, Copying(drafts=[script_draft(ids)]), n_variantes=1)
