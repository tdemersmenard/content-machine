"""Contrôle des preuves par le code."""

from typing import Any

from app.analysis.evidence import audit, build_index, check_evidence, iter_claims, tokens
from app.analysis.schemas import Evidence, ViralAnalysis
from tests.analysis_helpers import claim, quote, sample_analysis


def timeline() -> dict[str, Any]:
    words = [
        ("Today", 0.5),
        ("I", 0.8),
        ("will", 1.1),
        ("show", 1.4),
        ("you", 1.7),
        ("how", 2.0),
        ("to", 2.3),
        ("double", 2.6),
        ("your", 2.9),
        ("views", 3.2),
        ("l'argent,", 7.0),
    ]
    return {
        "video": {"duration_s": 9.0},
        "signals": {"shot_count": 3, "promise": None},
        "events": [
            {
                "type": "speech",
                "start_s": 0.5,
                "end_s": 7.25,
                "words": [{"word": w, "start": t, "end": t + 0.25} for w, t in words],
            },
            {"type": "frame", "start_s": 0.0, "on_screen_text": "DOUBLE TES VUES"},
            {"type": "frame", "start_s": 1.0, "on_screen_text": ""},
            {"type": "frame", "start_s": 8.0, "on_screen_text": "Le secret !"},
        ],
    }


METADATA = {"views": 100, "likes": None, "creator": "Ada", "hashtags": []}


def index():
    return build_index(timeline(), METADATA)


def evidence(kind: str, excerpt: str, start: float = 0, end: float = 9) -> Evidence:
    return Evidence(kind=kind, start_s=start, end_s=end, excerpt=excerpt)


def test_tokens_ignore_case_and_punctuation() -> None:
    assert tokens("L'argent, c'est TOUT !") == ["l", "argent", "c", "est", "tout"]


def test_exact_quote_is_valid() -> None:
    item = evidence("quote", "double your views", 2.6, 3.45)

    assert check_evidence(item, index()) == (True, None)


def test_quote_ignores_case_and_punctuation() -> None:
    assert check_evidence(evidence("quote", "DOUBLE your, views!", 2.6, 3.4), index())[0]


def test_quote_handles_apostrophes_inside_a_word() -> None:
    assert check_evidence(evidence("quote", "l'argent", 7, 7.25), index())[0]


def test_paraphrased_quote_is_rejected() -> None:
    valid, reason = check_evidence(evidence("quote", "double tes vues"), index())

    assert valid is False
    assert "introuvable" in reason


def test_quote_with_wrong_timestamp_is_repaired() -> None:
    item = evidence("quote", "double your views", 6.0, 6.5)

    valid, change = check_evidence(item, index())

    assert valid is True
    assert change
    assert (item.start_s, item.end_s) == (2.6, 3.45)


def test_screen_text_is_found_and_repaired() -> None:
    item = evidence("on_screen_text", "le secret", 0, 0)

    valid, change = check_evidence(item, index())

    assert valid is True
    assert change
    assert item.start_s == 8.0


def test_invented_screen_text_is_rejected() -> None:
    assert check_evidence(evidence("on_screen_text", "SOLDES -50 %"), index())[0] is False


def test_frame_needs_an_existing_moment() -> None:
    assert check_evidence(evidence("frame", "Un fond", 1, 1), index())[0] is True
    assert check_evidence(evidence("frame", "Un fond", 20, 20), index())[0] is False


def test_signal_must_exist_and_not_be_null() -> None:
    assert check_evidence(evidence("signal", "shot_count = 3"), index())[0] is True
    assert check_evidence(evidence("signal", "shot_count: 3"), index())[0] is True
    assert check_evidence(evidence("signal", "promise = oui"), index())[0] is False
    assert check_evidence(evidence("signal", "retention = 80 %"), index())[0] is False


def test_null_metadata_cannot_be_used_as_proof() -> None:
    assert check_evidence(evidence("metadata", "views = 100"), index())[0] is True
    assert check_evidence(evidence("metadata", "likes = 5000"), index())[0] is False
    assert check_evidence(evidence("metadata", "hashtags = #fyp"), index())[0] is False


def test_timestamp_after_the_end_is_rejected() -> None:
    valid, reason = check_evidence(evidence("signal", "shot_count = 3", 0, 60), index())

    assert valid is False
    assert "fin de la vidéo" in reason


def test_valid_analysis_is_left_untouched() -> None:
    data = sample_analysis()
    analysis = ViralAnalysis.model_validate(data)

    result = audit(analysis, index())

    assert result.removals == []
    assert result.corrections == []
    assert analysis == ViralAnalysis.model_validate(data)


def test_claim_without_valid_proof_is_removed() -> None:
    data = sample_analysis()
    data["hook"]["psychological_mechanism"] = [claim("Inventé", quote("jamais prononcé"))]
    analysis = ViralAnalysis.model_validate(data)

    result = audit(analysis, index())

    assert analysis.hook.psychological_mechanism == []
    assert [item.path for item in result.removals] == ["hook.psychological_mechanism[0]"]
    assert result.removals[0].statement == "Inventé"
    assert "introuvable" in result.removals[0].reason


def test_claim_keeps_its_valid_proofs_and_loses_the_others() -> None:
    data = sample_analysis()
    data["hook"]["visual"] = [claim("Mixte", quote(), quote("jamais prononcé"))]
    analysis = ViralAnalysis.model_validate(data)

    result = audit(analysis, index())

    assert len(analysis.hook.visual) == 1
    assert len(analysis.hook.visual[0].evidence) == 1
    assert result.removals == []


def test_list_item_is_dropped_when_its_main_claim_falls() -> None:
    data = sample_analysis()
    data["angle"]["alternatives"][1]["claim"] = claim("Faux", quote("jamais prononcé"))
    data["structure"]["beats"][0]["claim"] = claim("Faux", quote("jamais prononcé"))
    analysis = ViralAnalysis.model_validate(data)

    audit(analysis, index())

    assert [item.angle for item in analysis.angle.alternatives] == ["Angle 0", "Angle 2"]
    assert len(analysis.structure.beats) == 1


def test_loop_survives_when_only_its_closing_claim_falls() -> None:
    data = sample_analysis()
    data["retention"]["open_loops"][0]["closed"] = [claim("Faux", quote("jamais prononcé"))]
    analysis = ViralAnalysis.model_validate(data)

    audit(analysis, index())

    assert len(analysis.retention.open_loops) == 1
    assert analysis.retention.open_loops[0].closed == []


def test_beat_outside_the_video_is_removed() -> None:
    data = sample_analysis()
    data["structure"]["beats"][1]["end_s"] = 40
    analysis = ViralAnalysis.model_validate(data)

    result = audit(analysis, index())

    assert len(analysis.structure.beats) == 1
    assert "durée" in result.removals[0].reason


def test_no_claim_without_proof_survives_an_audit() -> None:
    data = sample_analysis()
    data["hook"]["spoken_text"] = [claim("Faux", quote("mots inventés"))]
    data["engagement"]["saves"] = [claim("Faux", quote("mots inventés"))]
    analysis = ViralAnalysis.model_validate(data)

    audit(analysis, index())

    assert all(item.evidence for item in iter_claims(analysis))
    assert "Faux" not in [item.statement for item in iter_claims(analysis)]
