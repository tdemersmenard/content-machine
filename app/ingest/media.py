"""Appels à ffmpeg et ffprobe."""

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

FRAME_MAX_WIDTH = 720  # Assez pour lire le texte à l'écran, sans gonfler le coût en tokens.

_PTS_TIME_RE = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")


class MediaError(Exception):
    """ffmpeg ou ffprobe a échoué, ou le fichier n'est pas une vidéo lisible."""


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    has_video: bool
    has_audio: bool


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except FileNotFoundError as exc:
        raise MediaError(f"{command[0]} est introuvable : est-il installé ?") from exc
    if result.returncode != 0:
        raise MediaError(f"{command[0]} a échoué : {result.stderr.strip()[-500:]}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> MediaInfo:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(result.stdout)
    stream_types = {stream.get("codec_type") for stream in data.get("streams", [])}
    try:
        duration_s = float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaError(f"Durée illisible pour {path.name}") from exc
    return MediaInfo(
        duration_s=duration_s,
        has_video="video" in stream_types,
        has_audio="audio" in stream_types,
    )


def extract_audio(video: Path, output: Path) -> None:
    """WAV mono 16 kHz, le format attendu par Whisper."""
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output),
        ]
    )


def parse_scene_times(ffmpeg_output: str) -> list[float]:
    return [float(value) for value in _PTS_TIME_RE.findall(ffmpeg_output)]


def detect_scene_cuts(video: Path, threshold: float) -> list[float]:
    result = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(video),
            "-an",
            "-filter:v",
            f"select='gt(scene,{threshold})',showinfo",
            "-f",
            "null",
            "-",
        ]
    )
    return parse_scene_times(result.stderr)


def extract_frame(video: Path, t_s: float, output: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{t_s:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            f"scale='min({FRAME_MAX_WIDTH},iw)':-2",
            "-q:v",
            "3",
            str(output),
        ]
    )
    if not output.is_file():
        raise MediaError(f"Aucune frame extraite à t={t_s:.3f}s")
