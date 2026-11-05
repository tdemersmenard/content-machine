"""Voix de l'utilisateur, tirée de ses propres vidéos.

Les mesures sont calculées par le code. Claude repère les tics et les expressions, et le code
ne garde que ceux qui figurent mot pour mot dans les transcriptions.
"""

import json
import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean, median

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.analysis.evidence import tokens
from app.config import get_settings
from app.db.models import Niche, Transcript, Video
from app.llm.client import ClaudeClient
from app.niche.profile import load_profile, set_profile, snapshot
from app.niche.schemas import VoiceProfile, VoiceTraits

logger = logging.getLogger(__name__)

SHORT_SENTENCE_WORDS = 5
MAX_PHRASES = 40

_SENTENCE_RE = re.compile(r"[.!?…]+")


@dataclass(frozen=True)
class VoiceStats:
    avg_sentence_words: float
    median_sentence_words: float
    short_sentence_ratio: float
    words_per_second: float
    silences_per_minute: float


def sentence_lengths(text: str) -> list[int]:
    return [count for part in _SENTENCE_RE.split(text) if (count := len(tokens(part)))]


def repeated_phrases(texts: list[str]) -> list[tuple[str, int]]:
    """Suites de 2 à 4 mots qui reviennent, les plus fréquentes d'abord."""
    counts: Counter[str] = Counter()
    for text in texts:
        words = tokens(text)
        for size in (2, 3, 4):
            counts.update(" ".join(words[i : i + size]) for i in range(len(words) - size + 1))
    repeated = [(phrase, count) for phrase, count in counts.most_common() if count >= 2]
    return repeated[:MAX_PHRASES]


def contains_phrase(texts: list[str], phrase: str) -> bool:
    needle = tokens(phrase)
    if not needle:
        return False
    size = len(needle)
    for text in texts:
        words = tokens(text)
        if any(words[i : i + size] == needle for i in range(len(words) - size + 1)):
            return True
    return False


def compute_stats(videos: list[Video], texts: list[str]) -> VoiceStats:
    lengths = [length for text in texts for length in sentence_lengths(text)]
    duration_s = sum(video.duration_s for video in videos)
    # Le débit de la timeline fait foi : tokens() compte « t'as » comme deux mots.
    word_count = sum(video.signals["words_per_second"] * video.duration_s for video in videos)
    silences = sum(len(video.signals["silences"]) for video in videos)
    return VoiceStats(
        avg_sentence_words=round(mean(lengths), 2),
        median_sentence_words=float(median(lengths)),
        short_sentence_ratio=round(
            sum(1 for length in lengths if length <= SHORT_SENTENCE_WORDS) / len(lengths), 3
        ),
        words_per_second=round(word_count / duration_s, 3),
        silences_per_minute=round(silences / (duration_s / 60), 3),
    )


def own_videos(session: Session, niche: Niche) -> list[tuple[Video, str]]:
    """Mes vidéos ingérées avec de la parole, pour cette niche ou sans niche précisée."""
    rows = session.execute(
        select(Video, Transcript.full_text)
        .join(Transcript, Transcript.video_id == Video.id)
        .where(
            Video.is_own_content.is_(True),
            Video.ingested_at.is_not(None),
            or_(Video.niche_source == niche.name, Video.niche_source.is_(None)),
        )
        .order_by(Video.id)
    ).all()
    return [(video, text) for video, text in rows if text.strip()]


def extract_traits(
    claude: ClaudeClient, texts: list[str], stats: VoiceStats, model: str
) -> tuple[VoiceTraits, str]:
    result = claude.call_json(
        prompt="voice_extract",
        version="v1",
        schema=VoiceTraits,
        model=model,
        variables={
            "transcripts": json.dumps(texts, ensure_ascii=False),
            "stats": json.dumps(vars(stats), ensure_ascii=False),
            "phrases": json.dumps(repeated_phrases(texts), ensure_ascii=False),
        },
        effort="medium",
    )
    traits = result.data
    return (
        VoiceTraits(
            verbal_tics=[item for item in traits.verbal_tics if contains_phrase(texts, item)],
            signature_expressions=[
                item for item in traits.signature_expressions if contains_phrase(texts, item)
            ],
            style_summary=traits.style_summary,
        ),
        result.prompt_version,
    )


def refresh_voice(
    session: Session, niche: Niche, claude: ClaudeClient, force: bool = False
) -> VoiceProfile | None:
    """Met à jour voice_profile. Sans effet si les vidéos prises en compte n'ont pas changé."""
    profile = load_profile(niche)
    rows = own_videos(session, niche)
    if not rows:
        return profile.voice_profile

    videos = [video for video, _ in rows]
    texts = [text for _, text in rows]
    video_ids = [video.id for video in videos]
    current = profile.voice_profile
    if current is not None and current.video_ids == video_ids and not force:
        return current

    stats = compute_stats(videos, texts)
    traits, prompt_version = extract_traits(claude, texts, stats, get_settings().model_generation)
    profile.voice_profile = VoiceProfile(
        video_ids=video_ids,
        **vars(stats),
        **traits.model_dump(),
        prompt_version=prompt_version,
    )
    set_profile(niche, profile)
    snapshot(session, niche, f"Voix mise à jour depuis {len(videos)} vidéo(s)")
    session.commit()
    return profile.voice_profile


def refresh_voice_for_video(
    session: Session, video_id: int, claude: Callable[[], ClaudeClient]
) -> None:
    """Appelée après l'ingestion. Un échec ne doit pas faire échouer l'ingestion."""
    video = session.get(Video, video_id)
    if video is None or not video.is_own_content:
        return

    statement = select(Niche).where(Niche.profile.is_not(None))
    if video.niche_source:
        statement = statement.where(Niche.name == video.niche_source)
    for niche in session.scalars(statement).all():
        try:
            refresh_voice(session, niche, claude())
        except Exception:
            logger.exception("Voix non mise à jour pour la niche %d", niche.id)
            session.rollback()
