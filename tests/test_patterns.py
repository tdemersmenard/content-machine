"""Extraction, fusion, recherche et exposition des patterns."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app.analysis.service import analyze, pending_video_ids
from app.api.patterns import get_embedder
from app.cli import cli
from app.config import get_settings
from app.db.models import (
    Analysis,
    Niche,
    Pattern,
    PatternType,
    Performance,
    Script,
    Video,
    VideoPattern,
)
from app.ingest.jobs import ingest
from app.llm.prompts import load_prompt
from app.main import app
from app.patterns.embeddings import embed_texts, from_bytes, to_bytes
from app.patterns.schemas import MergeDecision, PatternCandidate
from app.patterns.search import (
    score_mean,
    search_patterns,
    top_patterns,
    winning_combinations,
)
from app.patterns.service import extract_patterns
from tests.analysis_helpers import FakeAnalysisClaude, sample_analysis
from tests.ingest_helpers import make_deps, make_video
from tests.pattern_helpers import FakeEmbedder, FakePatternClaude, candidate


@pytest.fixture
def video_id(db: Session, sample_video: Path) -> int:
    return ingest(str(sample_video), deps=make_deps()).video.id


def new_analysis(db: Session, video_id: int) -> int:
    row = Analysis(
        video_id=video_id,
        prompt_version="analysis/v1",
        model="claude-opus-5-5",
        json_result={"analysis": sample_analysis()},
    )
    db.add(row)
    db.commit()
    return row.id


def run(db: Session, video_id: int, *candidates: dict, merges=None, embedder=None):
    fake = FakePatternClaude(list(candidates), merges)
    result = extract_patterns(db, new_analysis(db, video_id), fake, embedder or FakeEmbedder())
    return result, fake


def count(db: Session, model: type) -> int:
    return db.scalar(select(func.count()).select_from(model))


def add_pattern(db: Session, name: str, kind: str = "hook", **fields) -> Pattern:
    vector = FakeEmbedder()([name])[0]
    pattern = Pattern(
        type=kind,
        name=name,
        description=name,
        template=f"[{name}]",
        embedding=to_bytes(vector),
        embedding_model=get_settings().embedding_model,
        **fields,
    )
    db.add(pattern)
    db.commit()
    return pattern


def test_candidate_becomes_a_complete_pattern(db: Session, video_id: int) -> None:
    (pattern,), fake = run(db, video_id, candidate())

    assert pattern.type == "hook"
    assert pattern.name == "Promesse chiffrée"
    assert "[résultat]" in pattern.template
    assert pattern.when_to_use == "Quand le gain se mesure."
    assert pattern.when_not_to_use == "Quand le gain est incertain."
    assert (pattern.score_alpha, pattern.score_beta) == (1.0, 1.0)
    assert pattern.prompt_version == "pattern_extract/v1"
    assert from_bytes(pattern.embedding).shape == (256,)
    assert pattern.examples == [
        {"video_id": video_id, "excerpt": "double your views", "timestamp": 2.6}
    ]
    assert fake.count("pattern_merge") == 0


def test_video_is_linked_with_role_and_proof(db: Session, video_id: int) -> None:
    (pattern,), _ = run(db, video_id, candidate())

    link = db.scalars(select(VideoPattern)).one()

    assert (link.video_id, link.pattern_id, link.role) == (video_id, pattern.id, "opening")
    assert link.evidence == "[00:02.6] double your views"


def test_extraction_prompt_receives_the_analysis(db: Session, video_id: int) -> None:
    _, fake = run(db, video_id)

    assert "Il annonce le résultat" in fake.calls[0]["variables"]["analysis"]


def test_near_duplicate_of_same_mechanism_is_merged(
    db: Session, video_id: int, tmp_path: Path
) -> None:
    (first,), _ = run(db, video_id, candidate())
    before = first.embedding
    other = ingest(str(make_video(tmp_path / "b.mp4", "white")), deps=make_deps()).video.id

    (merged,), fake = run(db, other, candidate(), merges=[True])

    assert merged.id == first.id
    assert count(db, Pattern) == 1
    assert fake.count("pattern_merge") == 1
    assert merged.description == "Description enrichie du mécanisme."
    assert merged.embedding != before
    assert [item["video_id"] for item in merged.examples] == [video_id, other]
    assert count(db, VideoPattern) == 2


def test_near_duplicate_of_another_mechanism_creates_a_new_pattern(
    db: Session, video_id: int
) -> None:
    run(db, video_id, candidate())

    run(db, video_id, candidate(), merges=[False])

    assert count(db, Pattern) == 2


def test_distant_pattern_never_reaches_the_merge_prompt(db: Session, video_id: int) -> None:
    run(db, video_id, candidate("Promesse chiffrée"))

    _, fake = run(
        db,
        video_id,
        candidate(
            "Autre",
            description="Tout à fait différent sans aucun rapport.",
            template="Rien [ici]",
        ),
    )

    assert fake.count("pattern_merge") == 0
    assert count(db, Pattern) == 2


def test_only_patterns_of_the_same_type_are_compared(db: Session, video_id: int) -> None:
    run(db, video_id, candidate())

    _, fake = run(db, video_id, candidate(type="ending"))

    assert fake.count("pattern_merge") == 0
    assert count(db, Pattern) == 2


def test_threshold_is_strict_and_configurable(
    db: Session, video_id: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    run(db, video_id, candidate())
    monkeypatch.setenv("PATTERN_MERGE_THRESHOLD", "1.5")
    get_settings.cache_clear()

    _, fake = run(db, video_id, candidate())

    assert fake.count("pattern_merge") == 0


def test_next_neighbour_is_tried_when_the_closest_differs(db: Session, video_id: int) -> None:
    run(db, video_id, candidate())
    run(db, video_id, candidate(), merges=[False])

    patterns, fake = run(db, video_id, candidate(), merges=[False, True])

    assert fake.count("pattern_merge") == 2
    assert count(db, Pattern) == 2
    assert sum(item.description == "Description enrichie du mécanisme." for item in patterns) == 1


def test_two_identical_candidates_in_one_batch_are_compared(db: Session, video_id: int) -> None:
    _, fake = run(db, video_id, candidate(), candidate(), merges=[True])

    assert fake.count("pattern_merge") == 1
    assert count(db, Pattern) == 1


def test_candidate_without_valid_proof_is_ignored(db: Session, video_id: int) -> None:
    fake_proof = {"kind": "quote", "start_s": 1, "end_s": 2, "excerpt": "jamais prononcé"}

    patterns, _ = run(db, video_id, candidate(evidence=[fake_proof]))

    assert patterns == []
    assert count(db, Pattern) == 0


def test_only_valid_proofs_become_examples(db: Session, video_id: int) -> None:
    good = {"kind": "quote", "start_s": 2.6, "end_s": 3.45, "excerpt": "double your views"}
    bad = {"kind": "quote", "start_s": 1, "end_s": 2, "excerpt": "jamais prononcé"}

    (pattern,), _ = run(db, video_id, candidate(evidence=[good, bad]))

    assert [item["excerpt"] for item in pattern.examples] == ["double your views"]


def test_wrong_timestamp_is_repaired_in_the_example(db: Session, video_id: int) -> None:
    shifted = {"kind": "quote", "start_s": 7.0, "end_s": 7.5, "excerpt": "double your views"}

    (pattern,), _ = run(db, video_id, candidate(evidence=[shifted]))

    assert pattern.examples[0]["timestamp"] == pytest.approx(2.6)


def test_template_without_placeholder_is_rejected() -> None:
    with pytest.raises(ValidationError, match="trou"):
        PatternCandidate.model_validate(candidate(template="Une phrase sans trou"))


def test_merge_without_description_is_rejected() -> None:
    with pytest.raises(ValidationError, match="enrichie"):
        MergeDecision(same_mechanism=True, reasoning="x", merged_description=" ")
    assert MergeDecision(same_mechanism=False, reasoning="x", merged_description="")


def test_extraction_is_idempotent(db: Session, video_id: int) -> None:
    analysis_id = new_analysis(db, video_id)
    fake = FakePatternClaude([candidate()])
    embedder = FakeEmbedder()
    first = extract_patterns(db, analysis_id, fake, embedder)
    calls = len(fake.calls)

    second = extract_patterns(db, analysis_id, fake, embedder)

    assert [item.id for item in second] == [item.id for item in first]
    assert len(fake.calls) == calls


def test_same_example_is_not_added_twice(db: Session, video_id: int) -> None:
    (pattern,), _ = run(db, video_id, candidate())

    run(db, video_id, candidate(), merges=[True])

    assert len(pattern.examples) == 1
    assert count(db, VideoPattern) == 1


def test_resume_after_a_merge_failure_pays_nothing_twice(db: Session, video_id: int) -> None:
    run(db, video_id, candidate("Premier"), candidate("Deuxième"))
    analysis_id = new_analysis(db, video_id)
    failing = FakePatternClaude(
        [candidate("Nouveau", description="Sans rapport.", template="[x]"), candidate("Deuxième")],
        merges=[],
        fail_merge_at=0,
    )

    with pytest.raises(RuntimeError):
        extract_patterns(db, analysis_id, failing, FakeEmbedder())
    db.rollback()

    row = db.get(Analysis, analysis_id)
    assert row.pattern_candidates_done == 1
    assert row.patterns_extracted_at is None
    assert count(db, Pattern) == 3

    recovered = FakePatternClaude(merges=[True])
    extract_patterns(db, analysis_id, recovered, FakeEmbedder())

    assert recovered.count("pattern_extract") == 0
    assert recovered.count("pattern_merge") == 1
    assert db.get(Analysis, analysis_id).patterns_extracted_at is not None


def test_unknown_analysis_raises(db: Session) -> None:
    with pytest.raises(LookupError):
        extract_patterns(db, 999, FakePatternClaude(), FakeEmbedder())


class AnalysisThenPatterns(FakeAnalysisClaude):
    def __init__(self, pattern_claude) -> None:
        super().__init__()
        self.pattern_claude = pattern_claude

    def call_json(self, *, prompt, version, schema, model, **kwargs):
        if prompt.startswith("pattern"):
            return self.pattern_claude.call_json(
                prompt=prompt, version=version, schema=schema, model=model, **kwargs
            )
        return super().call_json(
            prompt=prompt, version=version, schema=schema, model=model, **kwargs
        )


def test_analyze_extracts_patterns_automatically(db: Session, video_id: int) -> None:
    fake = AnalysisThenPatterns(FakePatternClaude([candidate()]))

    row = analyze(db, video_id, fake, embedder=FakeEmbedder())

    assert row.patterns_extracted_at is not None
    assert db.scalars(select(Pattern)).one().name == "Promesse chiffrée"


def test_extraction_failure_keeps_the_analysis_and_retries_on_next_analyze(
    db: Session, video_id: int
) -> None:
    failing = FakeAnalysisClaude(fail_on="pattern_extract")

    row = analyze(db, video_id, failing)

    assert db.get(Analysis, row.id) is not None
    assert row.patterns_extracted_at is None
    assert pending_video_ids(db) == [video_id]

    fake = AnalysisThenPatterns(FakePatternClaude([candidate()]))
    again = analyze(db, video_id, fake, embedder=FakeEmbedder())

    assert again.id == row.id
    assert fake.count("analysis") == 0
    assert again.patterns_extracted_at is not None
    assert pending_video_ids(db) == []


def test_search_ranks_by_similarity(db: Session) -> None:
    add_pattern(db, "aveu erreur coûteuse")
    add_pattern(db, "chiffre choc surprenant")

    hits = search_patterns(db, "erreur coûteuse", embedder=FakeEmbedder())

    assert hits[0].name == "aveu erreur coûteuse"
    assert hits[0].similarity > hits[1].similarity


def test_search_filters_by_type_and_limits_results(db: Session) -> None:
    add_pattern(db, "un")
    add_pattern(db, "deux")
    add_pattern(db, "trois", "ending")

    embedder = FakeEmbedder()

    assert [h.name for h in search_patterns(db, "x", PatternType.ENDING, embedder=embedder)] == [
        "trois"
    ]
    assert len(search_patterns(db, "x", k=2, embedder=embedder)) == 2


def test_search_on_an_empty_library_does_not_load_the_model(db: Session) -> None:
    embedder = FakeEmbedder()

    assert search_patterns(db, "x", embedder=embedder) == []
    assert embedder.calls == 0


def test_search_ignores_vectors_from_another_model(db: Session) -> None:
    pattern = add_pattern(db, "ancien")
    pattern.embedding_model = "ancien-modele"
    pattern.embedding = to_bytes(np.zeros(8))
    db.commit()

    assert search_patterns(db, "x", embedder=FakeEmbedder()) == []


def test_score_is_the_mean_of_the_beta() -> None:
    assert score_mean(Pattern(score_alpha=1.0, score_beta=1.0)) == 0.5
    assert score_mean(Pattern(score_alpha=8.0, score_beta=2.0)) == 0.8


def test_top_patterns_rank_by_score_then_examples(db: Session) -> None:
    add_pattern(db, "moyen", examples=[{"video_id": 1, "timestamp": 0}])
    add_pattern(db, "fort", score_alpha=9.0)
    add_pattern(db, "moyen riche", examples=[{"video_id": 1, "timestamp": t} for t in (0, 1)])
    add_pattern(db, "autre type", "ending", score_alpha=20.0)

    ranked = top_patterns(db, PatternType.HOOK)

    assert [item.name for item in ranked] == ["fort", "moyen riche", "moyen"]


def test_top_patterns_filter_by_niche(db: Session) -> None:
    niche = Niche(name="fitness")
    video = Video(niche_source="fitness")
    other = Video(niche_source="cuisine")
    db.add_all([niche, video, other])
    db.commit()
    seen = add_pattern(db, "vu dans la niche")
    used = add_pattern(db, "utilisé en script")
    elsewhere = add_pattern(db, "ailleurs")
    db.add_all(
        [
            VideoPattern(video_id=video.id, pattern_id=seen.id, role="opening"),
            VideoPattern(video_id=other.id, pattern_id=elsewhere.id, role="opening"),
            Script(
                niche_id=niche.id,
                pattern_ids=[used.id],
                prompt_version="generate/v1",
                content={},
            ),
        ]
    )
    db.commit()

    names = {item.name for item in top_patterns(db, niche_id=niche.id)}

    assert names == {"vu dans la niche", "utilisé en script"}


def test_top_patterns_unknown_niche_raises(db: Session) -> None:
    with pytest.raises(LookupError):
        top_patterns(db, niche_id=999)


def library(db: Session, rows: list[tuple[int | None, list[Pattern]]]) -> list[Video]:
    videos = []
    for views, patterns in rows:
        video = Video(views=views)
        db.add(video)
        db.flush()
        db.add_all(
            [VideoPattern(video_id=video.id, pattern_id=p.id, role="opening") for p in patterns]
        )
        videos.append(video)
    db.commit()
    return videos


def test_combinations_favor_pairs_seen_in_the_best_videos(db: Session) -> None:
    a, b, c, d = (add_pattern(db, name) for name in "abcd")
    library(
        db,
        [
            (1_000_000, [a, b]),
            (900_000, [a, b]),
            (100, [c, d]),
            (50, [c, d]),
        ],
    )

    first, second = winning_combinations(db)

    assert first.pattern_names == ("a", "b")
    assert first.videos == 2
    assert first.weighted_score == pytest.approx(1.75)
    assert first.avg_percentile == pytest.approx(0.875)
    assert second.pattern_names == ("c", "d")
    assert second.weighted_score == pytest.approx(0.75)


def test_one_huge_video_does_not_outweigh_a_repeated_success(db: Session) -> None:
    a, b, c, d = (add_pattern(db, name) for name in "abcd")
    library(
        db,
        [
            (10**9, [a, b]),
            (1, [a, b]),
            (5000, [c, d]),
            (4000, [c, d]),
            (100, [a]),
        ],
    )

    assert winning_combinations(db)[0].pattern_names == ("c", "d")


def test_combinations_require_several_videos(db: Session) -> None:
    a, b = add_pattern(db, "a"), add_pattern(db, "b")
    library(db, [(100, [a, b])])

    assert winning_combinations(db) == []
    assert len(winning_combinations(db, min_videos=1)) == 1


def test_combinations_ignore_videos_without_views(db: Session) -> None:
    a, b = add_pattern(db, "a"), add_pattern(db, "b")
    library(db, [(None, [a, b]), (None, [a, b])])

    assert winning_combinations(db) == []


def test_combinations_use_the_latest_measure_of_own_videos(db: Session) -> None:
    a, b = add_pattern(db, "a"), add_pattern(db, "b")
    first, second = library(db, [(None, [a, b]), (None, [a, b])])
    for video, day, views in ((first, 1, 10), (first, 2, 500), (second, 1, 100)):
        db.add(
            Performance(
                video_id=video.id, measured_at=datetime(2026, 1, day, tzinfo=UTC), views=views
            )
        )
    db.commit()

    (combination,) = winning_combinations(db)

    assert combination.videos == 2
    assert combination.weighted_score == pytest.approx(1.5)


def test_lift_shows_when_a_pair_is_more_than_chance(db: Session) -> None:
    a, b, c = (add_pattern(db, name) for name in "abc")
    library(db, [(10, [a, b]), (20, [a, b]), (30, [c]), (40, [c])])

    assert winning_combinations(db)[0].lift == 2


client = TestClient(app)


@pytest.fixture
def api_embedder():
    app.dependency_overrides[get_embedder] = FakeEmbedder
    yield
    app.dependency_overrides.clear()


def test_get_patterns_lists_and_filters(db: Session) -> None:
    add_pattern(db, "un", score_alpha=5.0)
    add_pattern(db, "deux", "ending")

    assert [item["name"] for item in client.get("/patterns").json()] == ["un", "deux"]
    assert [item["name"] for item in client.get("/patterns?type=ending").json()] == ["deux"]
    assert client.get("/patterns?type=inconnu").status_code == 422
    assert client.get("/patterns?niche_id=999").status_code == 404


def test_get_pattern_shows_detail_without_the_embedding(db: Session, video_id: int) -> None:
    (pattern,), _ = run(db, video_id, candidate())

    body = client.get(f"/patterns/{pattern.id}").json()

    assert body["name"] == "Promesse chiffrée"
    assert body["score_mean"] == 0.5
    assert body["example_count"] == 1
    assert body["videos"][0]["video_id"] == video_id
    assert body["videos"][0]["role"] == "opening"
    assert "embedding" not in body
    assert client.get("/patterns/999").status_code == 404


def test_search_route_is_not_mistaken_for_an_id(db: Session, api_embedder) -> None:
    add_pattern(db, "aveu erreur")

    response = client.get("/patterns/search", params={"q": "aveu"})

    assert response.status_code == 200
    assert response.json()[0]["name"] == "aveu erreur"
    assert response.json()[0]["similarity"] > 0
    assert client.get("/patterns/search").status_code == 422


runner = CliRunner()


def test_cli_list(db: Session) -> None:
    add_pattern(db, "un hook")
    add_pattern(db, "une fin", "ending")

    result = runner.invoke(cli, ["patterns", "list", "--type", "hook"])

    assert result.exit_code == 0, result.output
    assert "un hook" in result.output
    assert "une fin" not in result.output
    assert runner.invoke(cli, ["patterns", "list", "--type", "faux"]).exit_code == 2


def test_cli_list_empty_library(db: Session) -> None:
    assert "Aucun pattern" in runner.invoke(cli, ["patterns", "list"]).output


def test_cli_show(db: Session, video_id: int) -> None:
    (pattern,), _ = run(db, video_id, candidate())

    result = runner.invoke(cli, ["patterns", "show", str(pattern.id)])

    assert result.exit_code == 0, result.output
    assert "Promesse chiffrée" in result.output
    assert "Quand l'éviter" in result.output
    assert "double your views" in result.output
    assert runner.invoke(cli, ["patterns", "show", "999"]).exit_code == 1


def test_prompts_render() -> None:
    assert "{{" not in load_prompt("pattern_extract", "v1").render_user({"analysis": "{}"})
    text = load_prompt("pattern_merge", "v1").render_user({"existing": "A", "candidate": "B"})
    assert "A" in text and "B" in text


@pytest.mark.embeddings
def test_real_embeddings_are_normalized_and_multilingual() -> None:
    vectors = embed_texts(
        [
            "Aveu d'erreur coûteuse : le créateur admet une erreur qui lui a coûté cher.",
            "Costly mistake confession: the creator admits an expensive mistake.",
            "Recette de gâteau au chocolat fondant pour le dessert.",
        ]
    )

    assert vectors.shape == (3, 384)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(1, abs=1e-4)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]


@pytest.mark.embeddings
def test_real_search_finds_a_pattern_from_a_query_in_another_language(db: Session) -> None:
    for name, text in (
        ("Aveu", "Le créateur admet une erreur qui lui a coûté cher."),
        ("Recette", "Montrer une recette de cuisine étape par étape."),
    ):
        db.add(
            Pattern(
                type="hook",
                name=name,
                description=text,
                embedding=to_bytes(embed_texts([text])[0]),
                embedding_model=get_settings().embedding_model,
            )
        )
    db.commit()

    hits = search_patterns(db, "admitting an expensive mistake")

    assert hits[0].name == "Aveu"
