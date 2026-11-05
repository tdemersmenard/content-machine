"""Entrevue d'onboarding : une question à la fois, puis synthèse du profil."""

import json
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import InterviewStatus, Niche, NicheInterview
from app.llm.client import ClaudeClient
from app.niche.profile import set_profile, snapshot
from app.niche.schemas import InterviewTurn, NicheProfile, ProfileDraft
from app.niche.topics import refresh_topics

logger = logging.getLogger(__name__)

MAX_ANSWERS = 40  # Garde-fou : une entrevue ne tourne pas indéfiniment.


class InterviewError(Exception):
    """L'entrevue ne peut pas recevoir cette action."""


def _transcript(interview: NicheInterview) -> str:
    return json.dumps(interview.transcript, ensure_ascii=False)


def _ask(interview: NicheInterview, claude: ClaudeClient) -> InterviewTurn:
    return claude.call_json(
        prompt="niche_interview",
        version="v1",
        schema=InterviewTurn,
        model=get_settings().model_generation,
        variables={"transcript": _transcript(interview)},
        effort="medium",
    ).data


def _unique_name(session: Session, wanted: str) -> str:
    taken = set(session.scalars(select(Niche.name)))
    name, suffix = wanted, 2
    while name in taken:
        name = f"{wanted} ({suffix})"
        suffix += 1
    return name


def finish_interview(
    session: Session, interview: NicheInterview, claude: ClaudeClient
) -> NicheInterview:
    """Synthétise le profil, crée la niche et sa banque de sujets."""
    if interview.status is InterviewStatus.DONE:
        return interview
    if not any(item["role"] == "user" for item in interview.transcript):
        raise InterviewError("Impossible de créer un profil sans aucune réponse.")

    draft = claude.call_json(
        prompt="niche_profile",
        version="v1",
        schema=ProfileDraft,
        model=get_settings().model_generation,
        variables={"transcript": _transcript(interview)},
        effort="high",
    ).data

    niche = Niche(name=_unique_name(session, interview.name or draft.niche_name), topics=[])
    profile = NicheProfile.model_validate(draft.model_dump())
    set_profile(niche, profile)
    session.add(niche)
    session.flush()
    niche.version = 0
    snapshot(session, niche, "Création par entrevue")
    interview.niche_id = niche.id
    interview.status = InterviewStatus.DONE
    session.commit()

    try:
        refresh_topics(session, niche, claude)
    except Exception:
        # Le profil est payé et enregistré : la banque de sujets se relance à part.
        logger.exception("Banque de sujets non générée pour la niche %d", niche.id)
        session.rollback()
    return interview


def start_interview(
    session: Session, claude: ClaudeClient, name: str | None = None
) -> NicheInterview:
    interview = NicheInterview(name=name, transcript=[], covered=[])
    turn = _ask(interview, claude)
    interview.transcript = [{"role": "assistant", "text": turn.next_question}]
    interview.covered = list(turn.covered_sections)
    session.add(interview)
    session.commit()
    return interview


def answer_interview(
    session: Session, interview: NicheInterview, answer: str, claude: ClaudeClient
) -> NicheInterview:
    """Enregistre une réponse et pose la question suivante, ou termine l'entrevue.

    Rien n'est enregistré si Claude échoue : il suffit de renvoyer la même réponse.
    """
    if interview.status is InterviewStatus.DONE:
        raise InterviewError("Cette entrevue est terminée.")
    if not answer.strip():
        raise InterviewError("La réponse est vide.")

    history = list(interview.transcript)
    interview.transcript = [*history, {"role": "user", "text": answer.strip()}]
    answers = sum(1 for item in interview.transcript if item["role"] == "user")

    try:
        if answers >= MAX_ANSWERS:
            return finish_interview(session, interview, claude)

        turn = _ask(interview, claude)
        interview.covered = list(turn.covered_sections)
        if turn.done:
            return finish_interview(session, interview, claude)
    except Exception:
        session.rollback()
        raise

    interview.transcript = [
        *interview.transcript,
        {"role": "assistant", "text": turn.next_question},
    ]
    session.commit()
    return interview


def current_question(interview: NicheInterview) -> str | None:
    if interview.status is InterviewStatus.DONE or not interview.transcript:
        return None
    last = interview.transcript[-1]
    return last["text"] if last["role"] == "assistant" else None
