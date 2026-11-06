"""Vidéo lisible par un navigateur.

TikTok et les iPhone livrent souvent du HEVC, que Chrome et Edge ne lisent pas partout. Une copie
H.264 est créée à la première lecture, à côté de la source. Les timestamps ne changent pas.
"""

import subprocess
import threading
from pathlib import Path

from app.config import get_settings
from app.db.models import Video

PLAYABLE = "h264"
_lock = threading.Lock()


class MediaError(Exception):
    """La vidéo ne peut pas être préparée."""


def _codec(path: Path) -> str:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def playable_path(video: Video) -> Path:
    if not video.file_path:
        raise MediaError("Cette vidéo n'a pas de fichier.")
    source = get_settings().data_dir.resolve() / video.file_path
    if not source.is_file():
        raise MediaError("Fichier vidéo introuvable.")
    if _codec(source) == PLAYABLE:
        return source

    preview = source.with_name("preview.mp4")
    with _lock:  # Le navigateur envoie plusieurs requêtes : une seule conversion à la fois.
        if preview.is_file():
            return preview
        partial = source.with_name("preview.part.mp4")
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(source),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "26",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(partial),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            partial.unlink(missing_ok=True)
            raise MediaError(f"Conversion échouée : {result.stderr[-300:]}")
        partial.replace(preview)
    return preview
