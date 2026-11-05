"""Transcription avec faster-whisper, timestamps au mot, langue auto-détectée."""

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.ingest.schemas import Transcription, Word

Transcriber = Callable[[Path], Transcription]


@lru_cache
def _load_model(name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device=device, compute_type=compute_type)


def transcribe_with_whisper(audio: Path, model_name: str | None = None) -> Transcription:
    settings = get_settings()
    model = _load_model(
        model_name or settings.whisper_model,
        settings.whisper_device,
        settings.whisper_compute_type,
    )
    # vad_filter limite les phrases hallucinées sur la musique et les silences.
    segments, info = model.transcribe(str(audio), word_timestamps=True, vad_filter=True)

    words: list[Word] = []
    texts: list[str] = []
    for segment in segments:
        texts.append(segment.text.strip())
        words.extend(
            Word(
                word=word.word.strip(),
                start=round(word.start, 3),
                end=round(word.end, 3),
                probability=round(word.probability, 3),
            )
            for word in segment.words or []
            if word.word.strip()
        )
    return Transcription(words=words, full_text=" ".join(texts), language=info.language)
