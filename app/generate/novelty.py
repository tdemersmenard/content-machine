"""Une idée déjà produite ne revient jamais : comparaison par embeddings.

La comparaison porte sur `core_idea` : l'idée en une phrase neutre, écrite par Claude en même temps
que l'idée. Comparer l'angle ne marche pas : mesuré sur de vraies idées, il reprend le gabarit du
hook (« Thomas ouvre sur... ») et le filtre compare alors le style, pas l'idée. Les scripts plus
anciens, sans `core_idea`, retombent sur leur angle et leur croyance attaquée.
"""

from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Script
from app.patterns.embeddings import Embedder, embed_texts, from_bytes, to_bytes

PAST_IDEAS_IN_PROMPT = 50


def idea_text(angle: str, belief_attacked: str) -> str:
    return f"{angle} {belief_attacked}".strip()


def _script_text(script: Script) -> str:
    content = script.content
    return content.get("core_idea") or idea_text(content["angle"], content["belief_attacked"])


@dataclass(frozen=True)
class Repeat:
    """Une idée écartée, et ce qu'elle répète. `script_id` vaut None dans un même lot."""

    text: str
    script_id: int | None
    similarity: float


@dataclass
class NoveltyFilter:
    """Se construit avant tout appel payant : si le modèle local manque, rien n'est dépensé."""

    session: Session
    niche_id: int
    embedder: Embedder | None = None
    past: list[str] = field(init=False)
    _known: list[tuple[int | None, np.ndarray]] = field(init=False)

    def __post_init__(self) -> None:
        self.embedder = self.embedder or embed_texts
        model = get_settings().embedding_model
        scripts = list(
            self.session.scalars(
                select(Script).where(Script.niche_id == self.niche_id).order_by(Script.id)
            )
        )
        stale = [item for item in scripts if item.embedding_model != model]
        if stale:
            vectors = self.embedder([_script_text(item) for item in stale])
            for item, vector in zip(stale, vectors, strict=True):
                item.idea_embedding = to_bytes(vector)
                item.embedding_model = model
            self.session.commit()

        self._known = [(item.id, from_bytes(item.idea_embedding)) for item in scripts]
        self.past = [_script_text(item) for item in scripts[-PAST_IDEAS_IN_PROMPT:]]

    def check(self, texts: list[str]) -> list[Repeat | None]:
        """Pour chaque texte, ce qu'il répète, ou None s'il est nouveau.

        Un texte nouveau rejoint les idées connues : deux idées d'un même lot ne passent pas.
        """
        if not texts:
            return []
        threshold = get_settings().idea_similarity_threshold
        results: list[Repeat | None] = []
        for text, vector in zip(texts, self.embedder(texts), strict=True):
            best_id, best_score = None, -1.0
            for script_id, known in self._known:
                score = float(known @ vector)
                if score > best_score:
                    best_id, best_score = script_id, score
            if best_score >= threshold:
                results.append(Repeat(text, best_id, round(best_score, 3)))
            else:
                results.append(None)
                self._known.append((None, vector))
        return results
