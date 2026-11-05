"""Acquisition par URL avec yt-dlp, et conversion de ses métadonnées.

Règle : une stat absente reste None. Rien n'est estimé ni remplacé par 0.
"""

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_HASHTAG_RE = re.compile(r"#(\w+)", re.UNICODE)


class AcquireError(Exception):
    """Téléchargement impossible."""


@dataclass(frozen=True)
class DownloadedVideo:
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_hashtags(text: str | None) -> list[str]:
    return list(dict.fromkeys(_HASHTAG_RE.findall(text or "")))


def _posted_at(info: dict[str, Any]) -> datetime | None:
    if info.get("timestamp") is not None:
        return datetime.fromtimestamp(info["timestamp"], UTC)
    if info.get("upload_date"):
        return datetime.strptime(info["upload_date"], "%Y%m%d").replace(tzinfo=UTC)
    return None


def _sound(info: dict[str, Any]) -> str | None:
    artist = info.get("artist") or next(iter(info.get("artists") or []), None)
    track = info.get("track")
    parts = [part for part in (artist, track) if part]
    return " - ".join(parts) or None


def metadata_from_info(info: dict[str, Any]) -> dict[str, Any]:
    """Convertit le dictionnaire yt-dlp vers les colonnes de la table videos."""
    description = info.get("description")
    tags = [tag for tag in info.get("tags") or [] if tag]
    return {
        "platform": (info.get("extractor_key") or "").lower() or None,
        "creator": info.get("uploader") or info.get("channel") or info.get("creator"),
        "creator_followers": info.get("channel_follower_count"),
        "posted_at": _posted_at(info),
        "views": info.get("view_count"),
        "likes": info.get("like_count"),
        "comments": info.get("comment_count"),
        "shares": info.get("repost_count"),
        "saves": info.get("save_count"),
        "description": description,
        "hashtags": list(dict.fromkeys([*extract_hashtags(description), *tags])),
        "sound": _sound(info),
    }


def download_url(url: str, destination: Path) -> DownloadedVideo:
    import yt_dlp

    options = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(destination / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
    except yt_dlp.utils.YoutubeDLError as exc:
        raise AcquireError(f"Téléchargement impossible : {exc}") from exc

    if info and info.get("entries"):
        info = next((entry for entry in info["entries"] if entry), None)
    files = sorted(destination.glob("source.*"))
    if not info or not files:
        raise AcquireError(f"Aucune vidéo récupérée depuis {url}")
    return DownloadedVideo(path=files[0], metadata=metadata_from_info(info))
