"""Vraie transcription : `uv run pytest -m whisper`. Télécharge le modèle tiny (~75 Mo) une fois."""

from pathlib import Path

import pytest

from app.ingest.media import extract_audio
from app.ingest.transcribe import transcribe_with_whisper


@pytest.mark.whisper
def test_real_transcription_gives_word_timestamps(sample_video: Path, tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    extract_audio(sample_video, audio)

    result = transcribe_with_whisper(audio, model_name="tiny")

    assert result.language == "en"
    assert "double" in result.full_text.lower()
    assert len(result.words) >= 8
    starts = [word.start for word in result.words]
    assert starts == sorted(starts)
    assert all(word.end >= word.start for word in result.words)
