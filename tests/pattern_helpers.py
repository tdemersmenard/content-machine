"""Doubles de test pour les patterns."""

import re
import zlib
from typing import Any

import numpy as np

from app.llm.client import ClaudeResult

DIMENSIONS = 256


class FakeEmbedder:
    """Sac de mots haché : deux textes qui partagent leurs mots ont un cosinus élevé."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        vectors = np.zeros((len(texts), DIMENSIONS), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                vectors[row, zlib.crc32(token.encode()) % DIMENSIONS] += 1
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors


def candidate(name: str = "Promesse chiffrée", **overrides: Any) -> dict[str, Any]:
    return {
        "type": "hook",
        "name": name,
        "description": f"{name} annonce un gain mesurable dès la première seconde.",
        "template": f"{name} : comment [résultat] en [durée]",
        "when_to_use": "Quand le gain se mesure.",
        "when_not_to_use": "Quand le gain est incertain.",
        "role": "opening",
        "evidence": [
            {"kind": "quote", "start_s": 2.6, "end_s": 3.45, "excerpt": "double your views"}
        ],
    } | overrides


class FakePatternClaude:
    """`merges` donne les réponses successives au prompt de fusion."""

    def __init__(
        self,
        candidates: list[dict] | None = None,
        merges: list[bool] | None = None,
        fail_merge_at: int | None = None,
    ) -> None:
        self.candidates = candidates or []
        self.merges = list(merges or [])
        self.fail_merge_at = fail_merge_at
        self.calls: list[dict[str, Any]] = []

    def call_json(self, *, prompt: str, version: str, schema: Any, model: str, **kwargs: Any):
        self.calls.append({"prompt": prompt, "version": version, **kwargs})
        if prompt == "pattern_extract":
            data = schema.model_validate({"candidates": self.candidates})
        else:
            if self.fail_merge_at is not None and self.count("pattern_merge") > self.fail_merge_at:
                raise RuntimeError("Panne simulée")
            same = self.merges.pop(0)
            data = schema(
                same_mechanism=same,
                reasoning="test",
                merged_description="Description enrichie du mécanisme." if same else "",
            )
        return ClaudeResult(
            data=data, model=model, prompt_version=f"{prompt}/{version}", attempts=1, cost_usd=0.01
        )

    def count(self, prompt: str) -> int:
        return sum(1 for call in self.calls if call["prompt"] == prompt)
