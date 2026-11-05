"""Timeline unifiée : le fichier que le module d'analyse lira.

Tous les événements (plans, parole, frames, silences, promesse) sont sur le même axe temps,
en secondes depuis le début de la vidéo, triés par `start_s`.
"""

import json
from pathlib import Path
from typing import Any

from app.db.models import Frame, Shot, Transcript, Video
from app.ingest.schemas import Signals, Word
from app.ingest.signals import SILENCE_MIN_S

TIMELINE_VERSION = 1

_TYPE_ORDER = {"shot": 0, "frame": 1, "promise": 2, "speech": 3, "silence": 4}


def group_speech(words: list[Word]) -> list[list[Word]]:
    """Regroupe les mots en passages, séparés par les silences."""
    groups: list[list[Word]] = []
    for word in words:
        if groups and word.start - groups[-1][-1].end <= SILENCE_MIN_S:
            groups[-1].append(word)
        else:
            groups.append([word])
    return groups


def build_timeline(
    video: Video,
    transcript: Transcript | None,
    shots: list[Shot],
    frames: list[Frame],
    signals: Signals,
) -> dict[str, Any]:
    words = [Word.model_validate(word) for word in transcript.words] if transcript else []

    events: list[dict[str, Any]] = [
        {
            "type": "shot",
            "start_s": shot.start_s,
            "end_s": shot.end_s,
            "index": shot.index,
            "on_screen_text": shot.on_screen_text,
            "visual_description": shot.visual_description,
        }
        for shot in shots
    ]
    events += [
        {
            "type": "frame",
            "start_s": frame.t_s,
            "end_s": frame.t_s,
            "shot_index": frame.shot_index,
            "reason": frame.reason,
            "on_screen_text": frame.on_screen_text,
            "visual_description": frame.visual_description,
            "framing": frame.framing,
            "face_present": frame.face_present,
            "camera_motion": frame.camera_motion,
            "graphic_elements": frame.graphic_elements,
        }
        for frame in frames
    ]
    events += [
        {
            "type": "speech",
            "start_s": group[0].start,
            "end_s": group[-1].end,
            "text": " ".join(word.word for word in group),
            "words": [word.model_dump(exclude={"probability"}) for word in group],
        }
        for group in group_speech(words)
    ]
    events += [
        {"type": "silence", "start_s": item.start_s, "end_s": item.end_s}
        for item in signals.silences
    ]
    if signals.promise:
        events.append(
            {
                "type": "promise",
                "start_s": signals.promise.start_s,
                "end_s": signals.promise.end_s,
                "text": signals.promise.text,
            }
        )
    events.sort(key=lambda event: (event["start_s"], _TYPE_ORDER[event["type"]]))

    return {
        "version": TIMELINE_VERSION,
        "video": {
            "id": video.id,
            "duration_s": video.duration_s,
            "platform": video.platform,
            "creator": video.creator,
            "posted_at": video.posted_at.isoformat() if video.posted_at else None,
            "is_own_content": video.is_own_content,
            "niche_source": video.niche_source,
            "language": transcript.language if transcript else None,
            "description": video.description,
            "hashtags": video.hashtags,
            "sound": video.sound,
            "stats": {
                "views": video.views,
                "likes": video.likes,
                "comments": video.comments,
                "shares": video.shares,
                "saves": video.saves,
            },
        },
        "full_text": transcript.full_text if transcript else "",
        "signals": signals.model_dump(),
        "events": events,
    }


def write_timeline(timeline: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
