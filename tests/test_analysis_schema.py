"""Garanties structurelles du schéma d'analyse."""

import json
import os
import typing
from pathlib import Path
from types import UnionType
from typing import Any

import pytest
from anthropic import transform_schema
from pydantic import BaseModel, ValidationError

from app.analysis.evidence import iter_claims
from app.analysis.schemas import (
    AnalysisDocument,
    Claim,
    DraftAnalysis,
    Evidence,
    PerformanceMetrics,
    Verification,
    VerifiedAnalysis,
    ViralAnalysis,
)
from app.niche.schemas import InterviewTurn, ProfileDraft, TopicBatch, VoiceTraits
from app.patterns.schemas import ExtractedPatterns, MergeDecision
from tests.analysis_helpers import sample_analysis

SNAPSHOT = Path(__file__).parent / "snapshots" / "analysis_schema.json"

# Texte libre autorisé hors d'un Claim : ce ne sont pas des affirmations sur la vidéo.
FREE_TEXT_ALLOWED = {
    ("Hook", "hook_type_other"),  # Libellé du type de hook.
    ("AlternativeAngle", "angle"),  # Idée alternative, hypothétique par nature.
    ("EmotionPoint", "emotion"),  # Libellé. L'affirmation est dans `claim`.
}


def walk_models(model: type[BaseModel], seen: set[type] | None = None) -> set[type[BaseModel]]:
    seen = seen if seen is not None else set()
    if model in seen:
        return seen
    seen.add(model)
    for field in model.model_fields.values():
        for inner in _types(field.annotation):
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                walk_models(inner, seen)
    return seen


def _types(annotation: Any) -> list[Any]:
    args = typing.get_args(annotation)
    return [annotation, *[inner for arg in args for inner in _types(arg)]]


def count_schema(node: Any) -> tuple[int, int]:
    """(paramètres optionnels, unions) d'un schéma, selon la définition de l'API."""
    optional = unions = 0
    if isinstance(node, dict):
        if "anyOf" in node or isinstance(node.get("type"), list):
            unions += 1
        if node.get("type") == "object" and "properties" in node:
            optional += len(set(node["properties"]) - set(node.get("required", [])))
        for value in node.values():
            more_optional, more_unions = count_schema(value)
            optional += more_optional
            unions += more_unions
    elif isinstance(node, list):
        for value in node:
            more_optional, more_unions = count_schema(value)
            optional += more_optional
            unions += more_unions
    return optional, unions


def test_schema_matches_snapshot() -> None:
    """Un changement de schéma casse les analyses stockées : il doit être délibéré."""
    schema = json.dumps(ViralAnalysis.model_json_schema(), indent=2, sort_keys=True) + "\n"

    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAPSHOT.write_text(schema, encoding="utf-8", newline="\n")

    assert schema == SNAPSHOT.read_text("utf-8"), "Relancer avec UPDATE_SNAPSHOTS=1 si voulu."


@pytest.mark.parametrize(
    "model",
    [
        DraftAnalysis,
        VerifiedAnalysis,
        ExtractedPatterns,
        MergeDecision,
        InterviewTurn,
        ProfileDraft,
        TopicBatch,
        VoiceTraits,
    ],
)
def test_schema_stays_under_the_api_limits(model: type[BaseModel]) -> None:
    """L'API refuse plus de 24 paramètres optionnels et 16 unions."""
    assert count_schema(transform_schema(model)) == (0, 0)


def test_schema_sent_to_the_api_has_no_recursion() -> None:
    assert '"$ref": "#"' not in json.dumps(transform_schema(VerifiedAnalysis))


def test_every_prose_field_is_a_claim() -> None:
    """Aucun texte libre ne peut contourner l'obligation de preuve."""
    for model in walk_models(ViralAnalysis) - {Claim, Evidence}:
        for name, field in model.model_fields.items():
            if field.annotation is str:
                assert (model.__name__, name) in FREE_TEXT_ALLOWED


def test_no_optional_field_in_what_claude_fills() -> None:
    for model in walk_models(ViralAnalysis):
        for name, field in model.model_fields.items():
            kinds = _types(field.annotation)
            assert not any(typing.get_origin(k) in (typing.Union, UnionType) for k in kinds), name
            assert field.is_required(), f"{model.__name__}.{name}"


def test_models_outside_lists_hold_no_lone_claim() -> None:
    """Un Claim seul ne se retire qu'avec son objet parent, qui doit être un élément de liste."""
    top_level = [
        ViralAnalysis,
        *[
            field.annotation
            for field in ViralAnalysis.model_fields.values()
            if isinstance(field.annotation, type)
        ],
    ]
    for model in top_level:
        for name, field in model.model_fields.items():
            assert field.annotation is not Claim, f"{model.__name__}.{name}"


def test_every_claim_of_a_valid_analysis_has_a_timestamp() -> None:
    claims = iter_claims(ViralAnalysis.model_validate(sample_analysis()))

    assert len(claims) > 20
    for claim in claims:
        assert claim.evidence
        for evidence in claim.evidence:
            assert evidence.start_s >= 0
            assert evidence.end_s >= evidence.start_s
            assert evidence.excerpt


def test_claim_without_evidence_is_rejected() -> None:
    data = sample_analysis()
    data["hook"]["visual"][0]["evidence"] = []

    with pytest.raises(ValidationError):
        ViralAnalysis.model_validate(data)


def test_evidence_without_timestamp_is_rejected() -> None:
    data = sample_analysis()
    del data["hook"]["visual"][0]["evidence"][0]["start_s"]

    with pytest.raises(ValidationError):
        ViralAnalysis.model_validate(data)


def test_evidence_with_reversed_timestamps_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Evidence(kind="frame", start_s=5, end_s=2, excerpt="x")


def test_score_must_stay_between_0_and_10() -> None:
    data = sample_analysis()
    data["hook"]["score"] = 11

    with pytest.raises(ValidationError):
        ViralAnalysis.model_validate(data)


def test_draft_requires_three_alternatives_and_three_ranked_reasons() -> None:
    assert DraftAnalysis.model_validate(sample_analysis())

    missing_alternative = sample_analysis()
    missing_alternative["angle"]["alternatives"].pop()
    with pytest.raises(ValidationError, match="3 angles"):
        DraftAnalysis.model_validate(missing_alternative)

    duplicate_rank = sample_analysis()
    duplicate_rank["verdict"]["reasons"][1]["rank"] = 1
    with pytest.raises(ValidationError, match="3 raisons"):
        DraftAnalysis.model_validate(duplicate_rank)


def test_verified_analysis_may_have_lost_elements() -> None:
    data = sample_analysis()
    data["angle"]["alternatives"] = []
    data["verdict"]["reasons"] = []

    assert ViralAnalysis.model_validate(data)


def test_stored_document_round_trips_through_json() -> None:
    document = AnalysisDocument(
        analysis=ViralAnalysis.model_validate(sample_analysis()),
        metrics=PerformanceMetrics(
            views_per_follower=None,
            share_rate=0.01,
            save_rate=None,
            comment_rate=None,
            like_rate=None,
        ),
        verification=Verification(
            prompt_version="analysis_verify/v1",
            model="claude-opus-5-5",
            llm_corrections=[],
            llm_removals=[],
            code_corrections=[],
            code_removals=[],
        ),
    )

    assert AnalysisDocument.model_validate(document.model_dump(mode="json")) == document
