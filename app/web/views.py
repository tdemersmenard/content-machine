"""Mise en forme des données pour les pages. Aucun appel à Claude."""

import re
from collections import defaultdict
from statistics import mean
from typing import Any

from markdown_it import MarkdownIt
from markupsafe import Markup, escape
from sqlalchemy.orm import Session

from app.analysis.report import ROLES, clock
from app.db.models import Pattern
from app.learning.scoring import SUCCESS_ABOVE
from app.learning.service import scored_results

MIN_RESULTS_FOR_TREND = 4

# html=False : une sortie de Claude ou un texte de vidéo ne peut pas injecter de HTML.
_markdown = MarkdownIt("commonmark", {"html": False})
_CITATION_RE = re.compile(r"\[(vidéo|pattern|leçon|script)\s+#?(\d+)\]", re.IGNORECASE)
_MOMENT_RE = re.compile(r"\[(\d{2}):(\d{2}\.\d)(?:–\d{2}:\d{2}\.\d)?\]")
_VERIFY_RE = re.compile(r"\[À VÉRIFIER[^\]]*\]")
_LINKS = {
    "vidéo": "/ui/videos/{}",
    "pattern": "/ui/patterns#pattern-{}",
    "leçon": "/ui/brain#lesson-{}",
    "script": "/ui/studio?ids={}",
}


def _citation(match: re.Match[str]) -> str:
    kind, value = match.group(1).lower(), match.group(2)
    return f'<a class="citation" href="{_LINKS[kind].format(value)}">{kind} {value}</a>'


def link_citations(html: str) -> str:
    return _CITATION_RE.sub(_citation, html)


def render_markdown(text: str) -> Markup:
    return Markup(link_citations(_markdown.render(text)))


def render_report_html(text: str) -> Markup:
    """Le rapport d'analyse, où chaque moment [00:12.4] saute dans la vidéo."""

    def seek(match: re.Match[str]) -> str:
        seconds = int(match.group(1)) * 60 + float(match.group(2))
        return f'<button type="button" class="seek" data-seek="{seconds}">{match.group(0)}</button>'

    return Markup(_MOMENT_RE.sub(seek, _markdown.render(text)))


def linked_text(text: str) -> Markup:
    """Texte brut échappé, avec les références aux vidéos et aux patterns cliquables."""
    return Markup(link_citations(str(escape(text))))


def highlight_verify(text: str) -> Markup:
    return Markup(_VERIFY_RE.sub(lambda match: f"<mark>{match.group(0)}</mark>", str(escape(text))))


def _start(claim: dict[str, Any]) -> tuple[float, float]:
    first = claim["evidence"][0]
    return first["start_s"], first["end_s"]


def moments(analysis: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    """Les points cliquables de la timeline : hook, beats, boucles, re-hooks et risques."""
    found: list[dict[str, Any]] = []

    def add(kind: str, label: str, start: float, end: float, text: str) -> None:
        found.append(
            {
                "kind": kind,
                "label": label,
                "start": start,
                "end": end,
                "clock": clock(start),
                "text": text,
                "percent": round(min(start / duration, 1) * 100, 2) if duration else 0,
            }
        )

    for claim in analysis["hook"]["spoken_text"][:1]:
        add("hook", f"Hook {analysis['hook']['score']}/10", *_start(claim), claim["statement"])
    for beat in analysis["structure"]["beats"]:
        role = beat["role"]
        if role != "hook":
            add(
                "beat",
                ROLES[role],
                beat["start_s"],
                beat["end_s"],
                beat["claim"]["statement"],
            )
    for loop in analysis["retention"]["open_loops"]:
        add("loop", "Boucle ouverte", *_start(loop["opened"]), loop["opened"]["statement"])
        for claim in loop["closed"]:
            add("loop", "Boucle fermée", *_start(claim), claim["statement"])
    for claim in analysis["retention"]["rehooks"]:
        add("rehook", "Re-hook", *_start(claim), claim["statement"])
    for risk in analysis["retention"]["drop_risks"]:
        add("risk", "Risque de décrochage", *_start(risk["risk"]), risk["risk"]["statement"])
    return sorted(found, key=lambda item: item["start"])


def confidence(pattern: Pattern) -> dict[str, Any]:
    """Score et force de la preuve. Sans test, le score de 0,5 n'est qu'un a priori."""
    tests = pattern.score_alpha + pattern.score_beta - 2
    return {
        "score": pattern.score_alpha / (pattern.score_alpha + pattern.score_beta),
        "tests": round(tests, 1),
        "tested": tests > 0,
        "bar": min(tests / 10, 1) * 100,  # Pleine à 10 résultats pondérés.
    }


def pattern_trends(session: Session, niche_id: int) -> dict[str, list[dict[str, Any]]]:
    """Taux de réussite récent contre ancien, sur au moins 4 résultats par pattern."""
    results = [item for item in scored_results(session) if item[0].niche_id == niche_id]
    results.sort(key=lambda item: item[1].measured_at)
    by_pattern: dict[int, list[bool]] = defaultdict(list)
    for script, row in results:
        for pattern_id in set(script.pattern_ids):
            by_pattern[pattern_id].append(row.composite > SUCCESS_ABOVE)

    trends = []
    for pattern_id, outcomes in by_pattern.items():
        if len(outcomes) < MIN_RESULTS_FOR_TREND:
            continue
        middle = len(outcomes) // 2
        change = mean(outcomes[middle:]) - mean(outcomes[:middle])
        pattern = session.get(Pattern, pattern_id)
        if pattern is not None and change:
            trends.append(
                {
                    "id": pattern_id,
                    "name": pattern.name,
                    "change": round(change * 100),
                    "results": len(outcomes),
                }
            )
    trends.sort(key=lambda item: item["change"], reverse=True)
    return {
        "rising": [item for item in trends if item["change"] > 0][:5],
        "falling": [item for item in reversed(trends) if item["change"] < 0][:5],
    }


def composite_chart(session: Session, niche_id: int) -> dict[str, Any] | None:
    """Une ligne, un point par vidéo notée. Échelle en SVG calculée ici."""
    results = [item for item in scored_results(session) if item[0].niche_id == niche_id]
    results.sort(key=lambda item: item[1].measured_at)
    if not results:
        return None

    width, height, left, right, top, bottom = 640, 240, 36, 12, 12, 26
    values = [row.composite for _, row in results]
    highest = max(*values, 2.0) * 1.1
    plot_w, plot_h = width - left - right, height - top - bottom

    def y(value: float) -> float:
        return round(top + plot_h * (1 - value / highest), 1)

    def x(index: int) -> float:
        if len(results) == 1:
            return left + plot_w / 2
        return round(left + plot_w * index / (len(results) - 1), 1)

    points = [
        {
            "x": x(index),
            "y": y(row.composite),
            "script_id": script.id,
            "title": script.content["internal_title"],
            "date": row.measured_at.strftime("%Y-%m-%d"),
            "composite": round(row.composite, 2),
            "label": row.label,
            "views": row.views,
        }
        for index, (script, row) in enumerate(results)
    ]
    step = 0.5 if highest <= 4 else 1.0 if highest <= 8 else 2.0
    ticks = [
        {"value": value, "y": y(value)}
        for value in [step * index for index in range(int(highest / step) + 1)]
    ]
    return {
        "width": width,
        "height": height,
        "left": left,
        "right": width - right,
        "bottom": height - bottom,
        "points": points,
        "line": " ".join(f"{item['x']},{item['y']}" for item in points),
        "baseline_y": y(1.0),
        "ticks": ticks,
        "first_date": points[0]["date"],
        "last_date": points[-1]["date"],
    }
