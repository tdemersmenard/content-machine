"""Signaux de rythme, calculés sur la transcription et les plans."""

import logging

from pydantic import BaseModel

from app.ingest.schemas import PromiseMoment, Signals, Silence, Word, WordRateBucket
from app.llm.client import ClaudeClient

logger = logging.getLogger(__name__)

BUCKET_S = 3.0
FIRST_SECONDS = 3.0
SILENCE_MIN_S = 0.4


class PromiseResult(BaseModel):
    found: bool
    promise: str | None
    start_word: int | None
    end_word: int | None


def find_silences(words: list[Word], duration_s: float) -> list[Silence]:
    """Passages sans parole de plus de 0,4 s. La musique de fond n'est pas prise en compte."""
    edges = [0.0, *[edge for word in words for edge in (word.start, word.end)], duration_s]
    silences = []
    for start, end in zip(edges[0::2], edges[1::2], strict=False):
        if end - start > SILENCE_MIN_S:
            silences.append(
                Silence(
                    start_s=round(start, 3), end_s=round(end, 3), duration_s=round(end - start, 3)
                )
            )
    return silences


def word_rate_buckets(words: list[Word], duration_s: float) -> list[WordRateBucket]:
    buckets = []
    start = 0.0
    while start < duration_s:
        end = min(start + BUCKET_S, duration_s)
        count = sum(1 for word in words if start <= word.start < end)
        buckets.append(
            WordRateBucket(
                start_s=round(start, 3),
                end_s=round(end, 3),
                words=count,
                words_per_second=round(count / (end - start), 3),
            )
        )
        start += BUCKET_S
    return buckets


def compute_signals(
    words: list[Word],
    shots: list[tuple[float, float]],
    duration_s: float,
    promise: PromiseMoment | None,
) -> Signals:
    cuts = [start for start, _ in shots[1:]]
    return Signals(
        words_per_second=round(len(words) / duration_s, 3),
        words_per_second_by_3s=word_rate_buckets(words, duration_s),
        shot_count=len(shots),
        avg_shot_duration_s=round(duration_s / len(shots), 3),
        cuts_first_3s=sum(1 for cut in cuts if cut < FIRST_SECONDS),
        silences=find_silences(words, duration_s),
        promise=promise,
    )


def find_promise(claude: ClaudeClient, words: list[Word], model: str) -> PromiseMoment | None:
    """Claude désigne des mots : les timestamps restent ceux de la transcription."""
    if not words:
        return None

    result = claude.call_json(
        prompt="promise",
        version="v1",
        schema=PromiseResult,
        model=model,
        variables={"words": [word.word for word in words]},
        effort="medium",
    ).data

    if not result.found or result.start_word is None or result.end_word is None:
        return None
    if not 0 <= result.start_word <= result.end_word < len(words):
        logger.warning("Indices de promesse hors limites : %s", result)
        return None

    passage = words[result.start_word : result.end_word + 1]
    return PromiseMoment(
        text=" ".join(word.word for word in passage),
        start_s=passage[0].start,
        end_s=passage[-1].end,
    )
