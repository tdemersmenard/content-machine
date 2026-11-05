"""Import CSV de stats.

Les préréglages couvrent les libellés courants, en français et en anglais, mais ces exports
changent souvent : `mapping` permet de préciser ou de remplacer n'importe quelle colonne.
"""

import csv
import io
import re
from dataclasses import dataclass, field
from typing import Any, get_args

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Script, Video
from app.learning.schemas import Horizon, PerformanceInput
from app.learning.service import rebuild_pattern_scores, save_performance

RATE_FIELDS = {"retention_3s", "retention_50", "completion_rate"}
FLOAT_FIELDS = {"avg_watch_time_s"}
COUNT_FIELDS = {"views", "likes", "shares", "saves", "comments", "follows"}
LINK_FIELDS = {"script_id", "video_url"}
ALL_FIELDS = RATE_FIELDS | FLOAT_FIELDS | COUNT_FIELDS | LINK_FIELDS

PRESETS: dict[str, dict[str, str]] = {
    "tiktok": {
        "video views": "views",
        "views": "views",
        "vues": "views",
        "likes": "likes",
        "comments": "comments",
        "commentaires": "comments",
        "shares": "shares",
        "partages": "shares",
        "saves": "saves",
        "favorites": "saves",
        "enregistrements": "saves",
        "average watch time": "avg_watch_time_s",
        "durée moyenne de visionnage": "avg_watch_time_s",
        "watched full video": "completion_rate",
        "new followers": "follows",
        "video link": "video_url",
        "lien de la vidéo": "video_url",
    },
    "meta": {
        "plays": "views",
        "views": "views",
        "vues": "views",
        "likes": "likes",
        "j'aime": "likes",
        "comments": "comments",
        "commentaires": "comments",
        "shares": "shares",
        "partages": "shares",
        "saves": "saves",
        "enregistrements": "saves",
        "average watch time": "avg_watch_time_s",
        "follows": "follows",
        "permalink": "video_url",
    },
    "youtube": {
        "views": "views",
        "vues": "views",
        "likes": "likes",
        "comments added": "comments",
        "commentaires ajoutés": "comments",
        "shares": "shares",
        "partages": "shares",
        "subscribers gained": "follows",
        "abonnés gagnés": "follows",
        "average view duration": "avg_watch_time_s",
        "durée moyenne de visionnage": "avg_watch_time_s",
        "average percentage viewed (%)": "completion_rate",
        "video link": "video_url",
    },
}


class ImportError_(Exception):
    """Le fichier ne peut pas être importé."""


@dataclass
class ImportResult:
    imported: int = 0
    skipped: list[str] = field(default_factory=list)


def parse_rate(raw: str) -> float:
    """« 45 % », « 0,45 » ou « 45 » donnent 0,45."""
    text = raw.replace("%", "").replace(",", ".").strip()
    value = float(text)
    return value / 100 if "%" in raw or value > 1 else value


def parse_seconds(raw: str) -> float:
    """« 1:05 », « 0:01:05 », « 12,4 » ou « 12 s »."""
    text = raw.strip().lower().rstrip("s").strip().replace(",", ".")
    if ":" in text:
        seconds = 0.0
        for part in text.split(":"):
            seconds = seconds * 60 + float(part)
        return seconds
    return float(text)


def parse_count(raw: str) -> int:
    """« 12 345 », « 12,345 » ou « 1.2K »."""
    text = re.sub(r"[\s ]", "", raw).lower()
    multiplier = {"k": 1_000, "m": 1_000_000}.get(text[-1:], 1)
    if multiplier > 1:
        return round(float(text[:-1].replace(",", ".")) * multiplier)
    return int(text.replace(",", "").replace(".", ""))


def _parse(field_name: str, raw: str) -> Any:
    if raw.strip().lower() in ("", "-", "--", "n/a", "na", "null"):
        return None
    if field_name in RATE_FIELDS:
        return parse_rate(raw)
    if field_name in FLOAT_FIELDS:
        return parse_seconds(raw)
    if field_name in COUNT_FIELDS:
        return parse_count(raw)
    if field_name == "script_id":
        return int(raw)
    return raw.strip()


def _reader(content: str) -> csv.DictReader:
    try:
        dialect = csv.Sniffer().sniff(content[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return csv.DictReader(io.StringIO(content), dialect=dialect)


def _columns(
    headers: list[str], preset: str | None, mapping: dict[str, str] | None
) -> dict[str, str]:
    known = {**PRESETS.get(preset or "", {}), **{k.lower(): v for k, v in (mapping or {}).items()}}
    unknown = set(known.values()) - ALL_FIELDS
    if unknown:
        raise ImportError_(f"Champs inconnus dans le mapping : {sorted(unknown)}")
    columns = {
        header: known[header.strip().lower()]
        for header in headers
        if header.strip().lower() in known
    }
    columns.update({header: "script_id" for header in headers if header.strip() == "script_id"})
    if not set(columns.values()) & LINK_FIELDS:
        raise ImportError_(
            "Aucune colonne pour relier une ligne à un script. Ajoute une colonne script_id, "
            "ou précise la colonne du lien de la vidéo avec un mapping."
        )
    return columns


def _find_script(session: Session, values: dict[str, Any]) -> Script | None:
    if values.get("script_id") is not None:
        return session.get(Script, values["script_id"])
    if not values.get("video_url"):
        return None
    video = session.scalar(select(Video).where(Video.source_url == values["video_url"]))
    if video is None:
        return None
    return session.scalar(select(Script).where(Script.published_video_id == video.id))


def import_csv(
    session: Session,
    content: str,
    horizon_h: int,
    preset: str | None = None,
    mapping: dict[str, str] | None = None,
) -> ImportResult:
    """Importe un CSV. Une ligne sans script correspondant est signalée, jamais devinée.

    Les post-mortems ne sont pas lancés ici : `cm learn` s'en occupe.
    """
    if horizon_h not in get_args(Horizon):
        raise ImportError_(f"Horizon invalide : {horizon_h}. Choisis 24, 72 ou 168.")
    if preset is not None and preset not in PRESETS:
        raise ImportError_(f"Préréglage inconnu : {preset}")
    reader = _reader(content.lstrip("﻿"))
    columns = _columns(reader.fieldnames or [], preset, mapping)

    result = ImportResult()
    for number, line in enumerate(reader, 2):
        try:
            values = {
                field_name: _parse(field_name, line.get(header) or "")
                for header, field_name in columns.items()
            }
            script = _find_script(session, values)
            if script is None:
                result.skipped.append(f"Ligne {number} : aucun script correspondant.")
                continue
            values.pop("script_id", None)
            data = PerformanceInput(horizon_h=horizon_h, **values)
        except (ValueError, ValidationError) as exc:
            result.skipped.append(f"Ligne {number} : valeur illisible ({exc}).")
            continue
        save_performance(session, script, data, rebuild=False)
        result.imported += 1

    rebuild_pattern_scores(session)
    session.commit()
    return result
