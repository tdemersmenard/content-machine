"""Rendu markdown d'un script. Aucun appel à Claude."""

from app.analysis.report import HOOK_TYPES, ROLES, clock
from app.generate.schemas import ScriptContent

SOURCE_LABELS = {
    "source_analysis": "Analyse source",
    "lesson": "Leçon",
    "pattern": "Pattern",
    "profile": "Profil",
}
DURATION_SOURCES = {
    "voice_profile": "ton débit mesuré",
    "source_video": "le débit de la vidéo source, tant que tu n'as pas de vidéos ingérées",
    "default": "un débit par défaut, tant que tu n'as pas de vidéos ingérées",
}


def render_script(content: ScriptContent) -> str:
    c = content
    last = c.final_critique
    lines = [
        f"# {c.internal_title}",
        "",
        f"**Angle :** {c.angle}",
        "",
        f"**Croyance attaquée :** {c.belief_attacked}",
        "",
        f"**Durée estimée :** {c.estimated_duration_s:.0f} s, à {c.words_per_second:.1f} mots/s "
        f"({DURATION_SOURCES[c.duration_source]})",
        "",
        f"**Score prédit :** {c.predicted_score.value:.0f}/100 "
        f"(hook {c.predicted_score.hook_score}/10, non calibré)",
    ]
    if c.to_verify:
        lines += ["", f"**À vérifier avant de tourner :** {', '.join(c.to_verify)}"]

    lines += ["", "## Hooks", ""]
    for number, hook in enumerate(c.hooks, 1):
        lines += [
            f"**Hook {number} ({HOOK_TYPES[hook.hook_type]})**",
            "",
            f"- Dit : {hook.spoken}",
            f"- Écran : {hook.on_screen_text or 'rien'}",
            f"- Visuel : {hook.visual}",
            "",
        ]

    lines += ["## Script", ""]
    for number, beat in enumerate(c.beats):
        lines += [
            f"### Beat {number} : {ROLES[beat.role]} "
            f"({clock(beat.target_start_s)}–{clock(beat.target_end_s)})",
            "",
            beat.spoken,
            "",
            f"- Écran : {beat.on_screen_text or 'rien'}",
            f"- Plan : {beat.visual}",
            f"- Jeu : {beat.performance_note}",
            "",
        ]

    lines += ["## Boucles ouvertes", ""]
    lines += [
        f"- {loop.description} (ouverte au beat {loop.opened_beat}, "
        f"fermée au beat {loop.closed_beat})"
        for loop in c.open_loops
    ] or ["- Aucune"]

    lines += ["", "## Fin", "", c.ending, "", f"**CTA :** {c.cta}", ""]
    lines += [
        "## Légende",
        "",
        c.caption,
        "",
        " ".join(f"#{tag.lstrip('#')}" for tag in c.hashtags),
    ]

    lines += ["", "## Pourquoi ces choix", ""]
    lines += [
        f"- **{item.choice}** ({SOURCE_LABELS[item.source]}) : {item.reason}"
        for item in c.rationale
    ]
    lines += [
        "",
        f"Patterns utilisés : {', '.join(f'#{item}' for item in c.pattern_ids) or 'aucun'}",
    ]

    lines += [
        "",
        "## Critique du viewer impatient",
        "",
        f"Note du hook : {last.hook_score}/10, après {len(c.critiques) - 1} réécriture(s).",
        "",
    ]
    lines += [f"- « {item.quote} » : {item.reason}" for item in last.swipe_points] or [
        "- Aucun point de décrochage repéré."
    ]
    return "\n".join(lines) + "\n"
