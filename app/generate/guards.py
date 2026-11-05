"""Garde-fous vérifiés par le code, indépendamment de ce que le prompt demande."""

import re
from dataclasses import dataclass, field
from typing import Self

from pydantic import model_validator

from app.analysis.evidence import tokens
from app.generate.schemas import ScriptDraft
from app.niche.voice import contains_phrase

MAX_COPIED_WORDS = 5
VERIFY_MARKER = "À VÉRIFIER"
SMALL_COUNT_MAX = 3

_MEASURE_RE = re.compile(r"\s*(?:\$|%|€|k\b)", re.IGNORECASE)
_VERIFY_NOTE_RE = re.compile(rf"\[{VERIFY_MARKER}\s*:?\s*([^\]]*)\]")

_NUMBER_RE = re.compile(r"\d+(?:[  .,]\d{3})*(?:[.,]\d+)?(k\b)?", re.IGNORECASE)
_BRACKET_RE = re.compile(r"(\[[^\]]*\])")


def ngrams(text: str, size: int) -> set[tuple[str, ...]]:
    words = tokens(text)
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def copied_runs(text: str, source: set[tuple[str, ...]], size: int) -> list[str]:
    """Suites de `size` mots du texte qui figurent aussi dans la source."""
    words = tokens(text)
    return [
        " ".join(words[i : i + size])
        for i in range(len(words) - size + 1)
        if tuple(words[i : i + size]) in source
    ]


def normalize_number(raw: str) -> str:
    value = raw.lower().replace(" ", "").replace(" ", "")
    multiplier = 1000 if value.endswith("k") else 1
    value = value.removesuffix("k")
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", value):
        value = re.sub(r"[.,]", "", value)
    value = value.replace(",", ".")
    number = float(value) * multiplier
    return str(int(number)) if number == int(number) else str(number)


def numbers_in(text: str) -> set[str]:
    found = set()
    for match in _NUMBER_RE.finditer(text):
        found.add(normalize_number(match.group()))
    return found


def _is_small_count(match: re.Match[str]) -> bool:
    """« 1 client » ou « les 2 » ne sont pas des affirmations chiffrées, « 3 % » ou « 2 $ » oui."""
    if float(normalize_number(match.group())) > SMALL_COUNT_MAX:
        return False
    return _MEASURE_RE.match(match.string, match.end()) is None


def verification_notes(texts: list[str]) -> list[str]:
    """Tout ce qui reste à vérifier, chiffres marqués par le code et notes écrites par Claude."""
    notes = [
        match.group(1).strip() or "non précisé"
        for text in texts
        for match in _VERIFY_NOTE_RE.finditer(text)
    ]
    return list(dict.fromkeys(notes))


def flag_numbers(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    """Marque les chiffres absents de `allowed`. Le texte entre crochets n'est pas touché."""
    flagged: list[str] = []

    def replace(match: re.Match[str]) -> str:
        raw = match.group()
        if normalize_number(raw) in allowed or _is_small_count(match):
            return raw
        flagged.append(raw)
        return f"[{VERIFY_MARKER} : {raw}]"

    parts = _BRACKET_RE.split(text)
    parts = [part if part.startswith("[") else _NUMBER_RE.sub(replace, part) for part in parts]
    return "".join(parts), flagged


def spoken_texts(draft: ScriptDraft) -> list[str]:
    """Tout ce qui sera dit ou écrit à l'écran et dans la légende."""
    return [
        *[text for hook in draft.hooks for text in (hook.spoken, hook.on_screen_text)],
        *[text for beat in draft.beats for text in (beat.spoken, beat.on_screen_text)],
        draft.ending,
        draft.cta,
        draft.caption,
    ]


def flag_script(draft: ScriptDraft, allowed: set[str]) -> list[str]:
    """Marque sur place les chiffres non autorisés. Renvoie ceux qui ont été marqués."""
    flagged: list[str] = []

    def clean(text: str) -> str:
        result, found = flag_numbers(text, allowed)
        flagged.extend(found)
        return result

    for hook in draft.hooks:
        hook.spoken = clean(hook.spoken)
        hook.on_screen_text = clean(hook.on_screen_text)
    for beat in draft.beats:
        beat.spoken = clean(beat.spoken)
        beat.on_screen_text = clean(beat.on_screen_text)
    draft.ending = clean(draft.ending)
    draft.cta = clean(draft.cta)
    draft.caption = clean(draft.caption)
    return flagged


@dataclass(frozen=True)
class ScriptGuard:
    source_runs: set[tuple[str, ...]] = field(default_factory=set)
    never_say: tuple[str, ...] = ()
    allowed_pattern_ids: frozenset[int] = frozenset()

    def violations(self, draft: ScriptDraft) -> list[str]:
        errors: list[str] = []
        if len(draft.hooks) != 3:
            errors.append(f"Il faut exactement 3 hooks : {len(draft.hooks)} reçus.")
        if not draft.beats:
            errors.append("Il faut au moins un beat.")
        for loop in draft.open_loops:
            if not all(0 <= i < len(draft.beats) for i in (loop.opened_beat, loop.closed_beat)):
                errors.append(f"Boucle hors des beats : {loop.description}")
        unknown = set(draft.pattern_ids) - self.allowed_pattern_ids
        if unknown:
            errors.append(f"pattern_ids inconnus : {sorted(unknown)}")

        texts = spoken_texts(draft)
        for text in texts:
            for run in copied_runs(text, self.source_runs, MAX_COPIED_WORDS + 1):
                errors.append(f"Plus de {MAX_COPIED_WORDS} mots repris de la source : « {run} »")
        for phrase in self.never_say:
            if contains_phrase(texts, phrase):
                errors.append(f"Formule interdite par le créateur : « {phrase} »")
        return errors

    def schema(self) -> type[ScriptDraft]:
        """Schéma dont la validation applique les garde-fous, pour que le retry les corrige."""
        guard = self

        class GuardedScript(ScriptDraft):
            @model_validator(mode="after")
            def _guarded(self) -> Self:
                errors = guard.violations(self)
                if errors:
                    raise ValueError(" ; ".join(errors))
                return self

        return GuardedScript
