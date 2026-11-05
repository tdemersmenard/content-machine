"""Découpage en plans et choix des frames clés."""

from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise

HOOK_END_S = 5.0  # Le hook est critique : une frame par seconde.
HOOK_STEP_S = 1.0
BODY_STEP_S = 3.0
MERGE_TOLERANCE_S = 0.25  # Une frame forcée trop proche d'un cut n'apporte rien.
EDGE_MARGIN_S = 0.05  # Évite de demander une frame après la fin de la vidéo.


@dataclass(frozen=True)
class PlannedFrame:
    t_s: float
    reason: str  # "scene", "hook" ou "interval"
    shot_index: int


def clean_cuts(cuts: list[float], duration_s: float) -> list[float]:
    """Trie les cuts, retire les doublons et ceux collés au début ou à la fin."""
    cleaned: list[float] = []
    for cut in sorted(cuts):
        if not EDGE_MARGIN_S < cut < duration_s - EDGE_MARGIN_S:
            continue
        if cleaned and cut - cleaned[-1] < 0.1:
            continue
        cleaned.append(cut)
    return cleaned


def build_shots(cuts: list[float], duration_s: float) -> list[tuple[float, float]]:
    return list(pairwise([0.0, *cuts, duration_s]))


def forced_timestamps(duration_s: float) -> list[float]:
    """0, 1, 2, 3, 4, 5 puis 8, 11, 14..."""
    times: list[float] = []
    t = 0.0
    while t < duration_s - EDGE_MARGIN_S:
        times.append(t)
        t += HOOK_STEP_S if t < HOOK_END_S else BODY_STEP_S
    return times


def plan_frames(cuts: list[float], duration_s: float) -> list[PlannedFrame]:
    """Frames à extraire, triées. `cuts` doit venir de clean_cuts."""

    def shot_index(t: float) -> int:
        return bisect_right(cuts, t)

    planned = [PlannedFrame(cut, "scene", shot_index(cut)) for cut in cuts]
    for t in forced_timestamps(duration_s):
        if any(abs(t - cut) < MERGE_TOLERANCE_S for cut in cuts):
            continue
        planned.append(PlannedFrame(t, "hook" if t <= HOOK_END_S else "interval", shot_index(t)))
    return sorted(planned, key=lambda frame: frame.t_s)
