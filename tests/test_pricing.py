import pytest

from app.llm.pricing import TokenUsage, compute_cost


def test_cost_includes_every_token_category() -> None:
    usage = TokenUsage(
        model="claude-opus-5-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )

    assert compute_cost([usage]) == pytest.approx(4 + 20 + 0.20 + 5)


def test_cost_sums_models_at_their_own_rates() -> None:
    usages = [
        TokenUsage(model="claude-opus-5-5", output_tokens=1_000_000),
        TokenUsage(model="claude-sonnet-5-5", output_tokens=1_000_000),
    ]

    assert compute_cost(usages) == pytest.approx(30)


def test_unknown_model_gives_no_cost_rather_than_a_wrong_one() -> None:
    usages = [
        TokenUsage(model="claude-opus-5-5", output_tokens=10),
        TokenUsage(model="modele-inconnu", output_tokens=10),
    ]

    assert compute_cost(usages) is None
