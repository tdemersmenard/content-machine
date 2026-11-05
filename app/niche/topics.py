"""Banque de sujets : 50 douleurs de l'avatar, classées par intensité émotionnelle estimée."""

import json
import uuid
from typing import Self

from pydantic import model_validator
from sqlalchemy.orm import Session

from app.analysis.evidence import tokens
from app.config import get_settings
from app.db.models import Niche
from app.llm.client import ClaudeClient
from app.niche.profile import load_profile, snapshot
from app.niche.schemas import TopicBatch

TARGET_TOPICS = 50
ACTIVE = "new"
TOPIC_STATUSES = ("new", "used", "retired")


def _batch_schema(expected: int) -> type[TopicBatch]:
    """Schéma exigeant le bon nombre de sujets, pour que le retry du wrapper corrige l'écart."""

    class ExactTopicBatch(TopicBatch):
        @model_validator(mode="after")
        def _count(self) -> Self:
            if len(self.topics) != expected:
                raise ValueError(
                    f"Il faut exactement {expected} sujets : {len(self.topics)} reçus."
                )
            return self

    return ExactTopicBatch


def _key(title: str) -> tuple[str, ...]:
    return tuple(tokens(title))


def refresh_topics(session: Session, niche: Niche, claude: ClaudeClient) -> list[dict]:
    """Complète la banque jusqu'à 50 sujets actifs, sans toucher à ceux qui existent."""
    active = [item for item in niche.topics if item["status"] == ACTIVE]
    missing = TARGET_TOPICS - len(active)
    if missing <= 0:
        return niche.topics

    profile = load_profile(niche)
    result = claude.call_json(
        prompt="niche_topics",
        version="v1",
        schema=_batch_schema(missing),
        model=get_settings().model_generation,
        variables={
            "count": missing,
            "profile": profile.model_dump_json(exclude={"voice_profile"}),
            "existing": json.dumps([item["title"] for item in niche.topics], ensure_ascii=False),
        },
        effort="medium",
    )

    known = {_key(item["title"]) for item in niche.topics}
    new = []
    for idea in result.data.topics:
        key = _key(idea.title)
        if key in known:
            continue
        known.add(key)
        new.append({"id": uuid.uuid4().hex[:8], "status": ACTIVE, **idea.model_dump()})

    niche.topics = sorted([*niche.topics, *new], key=lambda item: item["intensity"], reverse=True)
    snapshot(session, niche, f"{len(new)} sujet(s) ajouté(s)")
    session.commit()
    return niche.topics


def set_topic_status(session: Session, niche: Niche, topic_id: str, status: str) -> dict:
    if status not in TOPIC_STATUSES:
        raise ValueError(f"Statut inconnu : {status}")
    topics = [dict(item) for item in niche.topics]
    topic = next((item for item in topics if item["id"] == topic_id), None)
    if topic is None:
        raise LookupError(f"Sujet {topic_id} inconnu.")
    topic["status"] = status
    niche.topics = topics
    snapshot(session, niche, f"Sujet {topic_id} : {status}")
    session.commit()
    return topic
