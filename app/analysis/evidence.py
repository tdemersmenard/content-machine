"""Contrôle des preuves par le code, indépendant de l'avis de Claude.

Une citation est cherchée dans les mots réellement transcrits, un texte d'écran dans les frames,
un signal ou une métadonnée dans les données fournies. Un timestamp faux est corrigé si la
preuve existe ailleurs. Une affirmation qui n'a plus aucune preuve valable est retirée.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.analysis.schemas import Claim, Correction, Evidence, Removal, ViralAnalysis

QUOTE_TOLERANCE_S = 1.0
FRAME_TOLERANCE_S = 3.0  # Écart maximal entre deux frames forcées.
END_TOLERANCE_S = 0.5

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_KEY_RE = re.compile(r"^\s*([A-Za-z_][\w.]*)")


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class SpokenToken:
    token: str
    start_s: float
    end_s: float


@dataclass
class EvidenceIndex:
    """Ce que la vidéo contient réellement."""

    duration_s: float
    spoken: list[SpokenToken]
    screen_texts: list[tuple[float, list[str]]]
    visual_times: list[float]
    known_signals: set[str]
    known_metadata: set[str]


def build_index(timeline: dict[str, Any], metadata: dict[str, Any]) -> EvidenceIndex:
    spoken: list[SpokenToken] = []
    screen_texts: list[tuple[float, list[str]]] = []
    visual_times: list[float] = []

    for event in timeline["events"]:
        if event["type"] == "speech":
            for word in event["words"]:
                spoken += [
                    SpokenToken(token, word["start"], word["end"]) for token in tokens(word["word"])
                ]
        elif event["type"] == "frame":
            visual_times.append(event["start_s"])
            if event.get("on_screen_text"):
                screen_texts.append((event["start_s"], tokens(event["on_screen_text"])))

    return EvidenceIndex(
        duration_s=timeline["video"]["duration_s"],
        spoken=spoken,
        screen_texts=screen_texts,
        visual_times=visual_times,
        known_signals={key for key, value in timeline["signals"].items() if value is not None},
        known_metadata={key for key, value in metadata.items() if value not in (None, [], "")},
    )


def _find_sequence(haystack: list[str], needle: list[str]) -> list[int]:
    size = len(needle)
    return [i for i in range(len(haystack) - size + 1) if haystack[i : i + size] == needle]


def _overlaps(start: float, end: float, evidence: Evidence, tolerance: float) -> bool:
    return start <= evidence.end_s + tolerance and end >= evidence.start_s - tolerance


def _check_quote(evidence: Evidence, index: EvidenceIndex) -> tuple[bool, str | None]:
    needle = tokens(evidence.excerpt)
    words = [item.token for item in index.spoken]
    matches = _find_sequence(words, needle) if needle else []
    if not matches:
        return False, "citation introuvable dans la transcription"

    spans = [(index.spoken[i].start_s, index.spoken[i + len(needle) - 1].end_s) for i in matches]
    if any(_overlaps(start, end, evidence, QUOTE_TOLERANCE_S) for start, end in spans):
        return True, None

    start, end = min(spans, key=lambda span: abs(span[0] - evidence.start_s))
    change = f"{evidence.start_s}-{evidence.end_s} -> {start}-{end}"
    evidence.start_s, evidence.end_s = start, end
    return True, change


def _check_screen_text(evidence: Evidence, index: EvidenceIndex) -> tuple[bool, str | None]:
    needle = tokens(evidence.excerpt)
    times = [t for t, text in index.screen_texts if needle and _find_sequence(text, needle)]
    if not times:
        return False, "texte absent des frames"
    if any(_overlaps(t, t, evidence, FRAME_TOLERANCE_S) for t in times):
        return True, None

    t = min(times, key=lambda value: abs(value - evidence.start_s))
    change = f"{evidence.start_s}-{evidence.end_s} -> {t}-{t}"
    evidence.start_s, evidence.end_s = t, t
    return True, change


def _key(excerpt: str) -> str | None:
    match = _KEY_RE.match(excerpt)
    return match.group(1).split(".")[0] if match else None


def check_evidence(evidence: Evidence, index: EvidenceIndex) -> tuple[bool, str | None]:
    """Renvoie (valide, détail). Le détail décrit la correction, ou le motif du rejet."""
    if evidence.kind == "quote":
        valid, detail = _check_quote(evidence, index)
    elif evidence.kind == "on_screen_text":
        valid, detail = _check_screen_text(evidence, index)
    elif evidence.kind == "signal":
        valid = _key(evidence.excerpt) in index.known_signals
        detail = None if valid else f"signal inconnu ou nul : {evidence.excerpt}"
    elif evidence.kind == "metadata":
        valid = _key(evidence.excerpt) in index.known_metadata
        detail = None if valid else f"métadonnée inconnue ou absente : {evidence.excerpt}"
    else:
        valid = any(_overlaps(t, t, evidence, FRAME_TOLERANCE_S) for t in index.visual_times)
        detail = None if valid else "aucune frame à ce moment"

    if valid and evidence.end_s > index.duration_s + END_TOLERANCE_S:
        return False, f"timestamp après la fin de la vidéo ({evidence.end_s} s)"
    return valid, detail


@dataclass
class AuditResult:
    corrections: list[Correction] = field(default_factory=list)
    removals: list[Removal] = field(default_factory=list)


def _audit_claim(claim: Claim, path: str, index: EvidenceIndex, result: AuditResult) -> bool:
    kept: list[Evidence] = []
    reasons: list[str] = []
    for position, evidence in enumerate(claim.evidence):
        valid, detail = check_evidence(evidence, index)
        if valid:
            kept.append(evidence)
            if detail:
                result.corrections.append(
                    Correction(path=f"{path}.evidence[{position}]", change=detail)
                )
        else:
            reasons.append(detail or "preuve invalide")

    claim.evidence = kept
    if kept:
        return True
    result.removals.append(Removal(path=path, statement=claim.statement, reason="; ".join(reasons)))
    return False


def _audit_model(node: BaseModel, path: str, index: EvidenceIndex, result: AuditResult) -> bool:
    """Renvoie False si l'objet a perdu son affirmation principale ou sort de la vidéo."""
    end_s = getattr(node, "end_s", None)
    if end_s is not None and end_s > index.duration_s + END_TOLERANCE_S:
        result.removals.append(
            Removal(path=path, statement=str(node)[:200], reason="hors de la durée de la vidéo")
        )
        return False

    for name in type(node).model_fields:
        value = getattr(node, name)
        child = f"{path}.{name}" if path else name
        if isinstance(value, Claim):
            if not _audit_claim(value, child, index, result):
                return False
        elif isinstance(value, BaseModel):
            _audit_model(value, child, index, result)
        elif isinstance(value, list):
            kept = []
            for position, item in enumerate(value):
                item_path = f"{child}[{position}]"
                if isinstance(item, Claim):
                    keep = _audit_claim(item, item_path, index, result)
                elif isinstance(item, BaseModel):
                    keep = _audit_model(item, item_path, index, result)
                else:
                    keep = True
                if keep:
                    kept.append(item)
            setattr(node, name, kept)
    return True


def audit(analysis: ViralAnalysis, index: EvidenceIndex) -> AuditResult:
    """Corrige `analysis` sur place : timestamps réparés, affirmations sans preuve retirées."""
    result = AuditResult()
    _audit_model(analysis, "", index, result)
    return result


def iter_claims(node: Any) -> list[Claim]:
    """Toutes les affirmations d'un objet, à n'importe quelle profondeur."""
    if isinstance(node, Claim):
        return [node]
    if isinstance(node, BaseModel):
        return [
            claim for name in type(node).model_fields for claim in iter_claims(getattr(node, name))
        ]
    if isinstance(node, list):
        return [claim for item in node for claim in iter_claims(item)]
    return []
