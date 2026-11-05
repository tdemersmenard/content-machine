"""Score normalisé : fonctions pures, sans base de données."""

from statistics import median

BASELINE_WINDOW = 20
WINNER_ABOVE = 1.5
LOSER_BELOW = 0.7
SUCCESS_ABOVE = 1.0

METRICS = (
    "views",
    "retention_3s",
    "retention_50",
    "completion_rate",
    "avg_watch_time_s",
    "shares",
    "saves",
    "comments",
    "follows",
    "likes",
)

# La rétention et les partages pèsent plus que les likes, comme demandé.
DEFAULT_WEIGHTS = {
    "retention_3s": 3.0,
    "retention_50": 3.0,
    "completion_rate": 2.0,
    "shares": 3.0,
    "saves": 2.0,
    "avg_watch_time_s": 1.0,
    "follows": 1.0,
    "comments": 1.0,
    "views": 1.0,
    "likes": 0.5,
}
GOAL_WEIGHTS = {
    "followers": {"follows": 4.0},
    "leads": {"comments": 2.0, "saves": 3.0, "follows": 2.0},
    "sales": {"saves": 3.0, "comments": 2.0},
    "authority": {"saves": 3.0, "completion_rate": 3.0},
}


def weights_for(goal: str | None, overrides: dict[str, float] | None = None) -> dict[str, float]:
    return {**DEFAULT_WEIGHTS, **GOAL_WEIGHTS.get(goal or "", {}), **(overrides or {})}


def baselines(history: list[dict[str, float | None]], minimum: int) -> dict[str, float]:
    """Médiane par métrique sur les relevés fournis, du plus ancien au plus récent.

    Seuls les 20 derniers comptent. Une métrique trop peu renseignée n'a pas de baseline.
    """
    recent = history[-BASELINE_WINDOW:]
    result = {}
    for metric in METRICS:
        values = [row[metric] for row in recent if row.get(metric) is not None]
        if len(values) >= minimum:
            result[metric] = median(values)
    return result


def performance_indexes(
    values: dict[str, float | None], baseline: dict[str, float]
) -> dict[str, float]:
    """valeur / baseline. Une métrique absente, ou de baseline nulle, n'a pas d'indice."""
    return {
        metric: round(values[metric] / reference, 4)
        for metric, reference in baseline.items()
        if values.get(metric) is not None and reference > 0
    }


def composite_score(indexes: dict[str, float], weights: dict[str, float]) -> float | None:
    """Moyenne pondérée des indices disponibles. Une stat manquante n'est jamais comptée à 0."""
    used = {metric: weights.get(metric, 0.0) for metric in indexes}
    total = sum(used.values())
    if total <= 0:
        return None
    return round(sum(indexes[metric] * weight for metric, weight in used.items()) / total, 4)


def classify(composite: float) -> str:
    if composite > WINNER_ABOVE:
        return "winner"
    if composite < LOSER_BELOW:
        return "loser"
    return "neutral"


def decay_weight(age_days: float, half_life_days: float) -> float:
    """1 pour un résultat d'aujourd'hui, 0,5 après une demi-vie."""
    return 0.5 ** (max(age_days, 0.0) / half_life_days)
