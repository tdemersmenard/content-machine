"""Tarifs Anthropic et calcul du coût d'un appel.

Prix en USD par million de tokens, à mettre à jour si la grille change.
L'écriture cache correspond au TTL de 5 minutes (1,25x le prix d'entrée).
Limite connue : le palier Haiku 5.5 au-delà de 100K tokens de prompt n'est pas modélisé.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float
    cache_read: float

    @property
    def cache_write(self) -> float:
        return self.input * 1.25


PRICES: dict[str, ModelPrice] = {
    "claude-fable-5-1": ModelPrice(10.00, 50.00, 0.25),
    "claude-opus-5-5": ModelPrice(4.00, 20.00, 0.20),
    "claude-opus-5": ModelPrice(5.00, 25.00, 0.50),
    "claude-opus-4-8": ModelPrice(5.00, 25.00, 0.50),
    "claude-sonnet-5-5": ModelPrice(2.00, 10.00, 0.20),
    "claude-sonnet-5": ModelPrice(2.00, 10.00, 0.20),
    "claude-haiku-5-5": ModelPrice(0.10, 0.50, 0.01),
}


@dataclass(frozen=True)
class TokenUsage:
    """Tokens consommés par un modèle au cours d'un appel."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


def compute_cost(usages: list[TokenUsage]) -> float | None:
    """Coût total en USD, ou None si un des modèles n'a pas de tarif connu."""
    total = 0.0
    for usage in usages:
        price = PRICES.get(usage.model)
        if price is None:
            logger.warning("Tarif inconnu pour le modèle %s : coût non calculé", usage.model)
            return None
        total += (
            usage.input_tokens * price.input
            + usage.output_tokens * price.output
            + usage.cache_read_tokens * price.cache_read
            + usage.cache_write_tokens * price.cache_write
        ) / 1_000_000
    return total
