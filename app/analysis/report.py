"""Rapport markdown lisible, généré à partir du JSON. Aucun appel à Claude."""

from app.analysis.schemas import AnalysisDocument, Claim, Evidence

HOOK_TYPES = {
    "contrarian": "Contrarien",
    "result_first": "Résultat d'abord",
    "question": "Question",
    "curiosity_gap": "Curiosity gap",
    "visual_pattern_interrupt": "Pattern interrupt visuel",
    "identification": "Identification (« si tu es X »)",
    "fear_loss": "Peur / perte",
    "shock_number": "Chiffre choc",
    "in_medias_res": "Histoire in medias res",
    "demonstration": "Démonstration",
    "other": "Autre",
}
ALIGNMENT = {
    "same_message": "les trois disent la même chose",
    "complementary": "ils se complètent",
    "contradictory": "ils se contredisent",
}
ROLES = {
    "hook": "Hook",
    "setup": "Setup",
    "tension": "Tension",
    "proof": "Preuve",
    "twist": "Retournement",
    "payoff": "Payoff",
    "cta": "CTA",
    "transition": "Transition",
    "other": "Autre",
}
FRAMEWORKS = {
    "pas": "PAS (problème, agitation, solution)",
    "before_after": "Avant / après",
    "list": "Liste",
    "story_loop": "Story loop",
    "tutorial": "Tutoriel",
    "myth_busting": "Démontage de mythe",
    "demonstration": "Démonstration",
    "hybrid": "Hybride",
}
ENDINGS = {
    "replay_loop": "Boucle de replay",
    "payoff": "Payoff",
    "cta": "CTA",
    "abrupt": "Fin abrupte",
    "mixed": "Mixte",
}
COMPARISONS = {
    "original_better": "l'original est meilleur",
    "original_worse": "l'original est moins bon",
    "equivalent": "équivalent",
}
COMMENT_TRIGGERS = {
    "debate": "Débat",
    "identification": "Identification",
    "open_question": "Question laissée ouverte",
    "deliberate_mistake": "Erreur volontaire",
    "other": "Autre",
}
SHARE_TRIGGERS = {
    "thats_so_you": "« C'est tellement toi »",
    "utility": "Utilité",
    "status": "Statut",
    "outrage": "Indignation",
    "other": "Autre",
}
CREATOR_SOURCES = {
    "charisma": "Charisme",
    "existing_audience": "Audience existante",
    "news_timing": "Timing d'actualité",
    "trending_sound": "Son tendance",
    "other": "Autre",
}
CONFIDENCE = {"low": "faible", "medium": "moyenne", "high": "élevée"}
VIDEO_KINDS = {
    "utility": "utilité",
    "entertainment": "divertissement",
    "debate": "débat",
    "mixed": "mixte",
    "unknown": "indéterminé",
}
STANCES = {"attacks": "Attaque", "confirms": "Confirme"}


def clock(seconds: float) -> str:
    tenths = round(seconds * 10)  # Arrondi avant découpage : 59,96 s donne 01:00.0.
    return f"{tenths // 600:02d}:{(tenths % 600) / 10:04.1f}"


def _proof(evidence: Evidence) -> str:
    moment = clock(evidence.start_s)
    if evidence.end_s > evidence.start_s:
        moment += f"–{clock(evidence.end_s)}"
    return f"[{moment}] {evidence.excerpt}"


def claim_line(claim: Claim, prefix: str = "") -> str:
    proofs = " · ".join(_proof(evidence) for evidence in claim.evidence)
    return f"- {prefix}{claim.statement}\n  > {proofs}"


def _claims(claims: list[Claim]) -> list[str]:
    return [claim_line(claim) for claim in claims] or ["- *Rien d'étayé sur ce point.*"]


def _percent(value: float | None) -> str:
    return "non disponible" if value is None else f"{value * 100:.2f} %"


def render_report(document: AnalysisDocument) -> str:
    a = document.analysis
    metrics = document.metrics
    verification = document.verification
    hook_type = HOOK_TYPES[a.hook.hook_type]
    if a.hook.hook_type_other:
        hook_type += f" : {a.hook.hook_type_other}"

    lines = ["# Pourquoi cette vidéo a marché", "", "## Verdict", ""]
    for reason in sorted(a.verdict.reasons, key=lambda item: item.rank):
        prefix = (
            f"**#{reason.rank}** ({reason.weight_percent} % du succès estimé, "
            f"confiance {CONFIDENCE[reason.confidence]}) : "
        )
        lines.append(claim_line(reason.claim, prefix))
    if not a.verdict.reasons:
        lines.append("- *Aucune raison n'a survécu à la vérification.*")

    lines += ["", f"## Hook : {a.hook.score}/10", "", f"**Type :** {hook_type}", ""]
    lines += ["**Texte dit**", "", *_claims(a.hook.spoken_text), ""]
    lines += ["**Texte à l'écran**", "", *_claims(a.hook.on_screen_text), ""]
    lines += ["**Visuel**", "", *_claims(a.hook.visual), ""]
    lines += ["**Pourquoi ce type**", "", *_claims(a.hook.hook_type_justification), ""]
    lines += ["**Pourquoi le cerveau ne swipe pas**", ""]
    lines += [*_claims(a.hook.psychological_mechanism), ""]
    lines += [f"**Alignement :** {ALIGNMENT[a.hook.alignment]}", ""]
    lines += [*_claims(a.hook.alignment_explanation), ""]
    lines += ["**Justification de la note**", "", *_claims(a.hook.score_justification)]

    lines += ["", "## Angle", "", "**Sujet brut**", "", *_claims(a.angle.raw_topic), ""]
    lines += ["**Angle choisi**", "", *_claims(a.angle.chosen_angle), ""]
    lines += ["**Croyances de l'audience**", ""]
    lines += [
        claim_line(belief.claim, f"{STANCES[belief.stance]} : ")
        for belief in a.angle.audience_beliefs
    ] or ["- *Rien d'étayé sur ce point.*"]
    lines += ["", "**Promesse implicite**", "", *_claims(a.angle.implicit_promise), ""]
    lines += ["**Moment où elle est payée**", "", *_claims(a.angle.promise_payoff), ""]
    lines += ["**Angles alternatifs**", ""]
    lines += [
        claim_line(item.claim, f"*{item.angle}* ({COMPARISONS[item.comparison]}) : ")
        for item in a.angle.alternatives
    ] or ["- *Rien d'étayé sur ce point.*"]

    lines += ["", "## Structure", "", f"**Framework :** {FRAMEWORKS[a.structure.framework]}", ""]
    lines += [*_claims(a.structure.framework_justification), "", "**Beats**", ""]
    lines += [
        claim_line(
            beat.claim, f"{clock(beat.start_s)}–{clock(beat.end_s)} **{ROLES[beat.role]}** : "
        )
        for beat in a.structure.beats
    ] or ["- *Rien d'étayé sur ce point.*"]

    lines += ["", "## Rétention", "", "**Boucles ouvertes**", ""]
    for loop in a.retention.open_loops:
        lines.append(claim_line(loop.opened, "Ouverture : "))
        lines += [claim_line(claim, "Fermeture : ") for claim in loop.closed]
        if not loop.closed:
            lines.append("- *Cette boucle ne se ferme pas.*")
    if not a.retention.open_loops:
        lines.append("- *Rien d'étayé sur ce point.*")
    lines += ["", "**Re-hooks**", "", *_claims(a.retention.rehooks), ""]
    lines += ["**Zones de risque**", ""]
    for risk in a.retention.drop_risks:
        lines.append(claim_line(risk.risk, "Risque : "))
        lines += [claim_line(claim, "Compensé par : ") for claim in risk.compensation]
    if not a.retention.drop_risks:
        lines.append("- *Rien d'étayé sur ce point.*")
    lines += ["", "**Rythme**", "", *_claims(a.retention.rhythm), ""]
    lines += [f"**Fin :** {ENDINGS[a.retention.ending_type]}", ""]
    lines += _claims(a.retention.ending_effect)

    lines += ["", "## Engagement", "", "**Pourquoi on commente**", ""]
    lines += [
        claim_line(item.claim, f"{COMMENT_TRIGGERS[item.trigger]} : ")
        for item in a.engagement.comments
    ] or ["- *Rien d'étayé sur ce point.*"]
    lines += ["", "**Pourquoi on partage**", ""]
    lines += [
        claim_line(item.claim, f"{SHARE_TRIGGERS[item.trigger]} : ") for item in a.engagement.shares
    ] or ["- *Rien d'étayé sur ce point.*"]
    lines += ["", "**Pourquoi on sauvegarde**", "", *_claims(a.engagement.saves)]

    lines += ["", "## Courbe émotionnelle", ""]
    lines += [
        claim_line(
            point.claim,
            f"{clock(point.start_s)}–{clock(point.end_s)} **{point.emotion}** "
            f"({point.intensity}/10) : ",
        )
        for point in a.emotion_curve
    ] or ["- *Rien d'étayé sur ce point.*"]

    lines += ["", "## Ce qui se transfère", "", "**Vient du format (réplicable)**", ""]
    lines += [*_claims(a.transferability.format_elements), ""]
    lines += ["**Vient du créateur (à ne pas copier)**", ""]
    lines += [
        claim_line(item.claim, f"{CREATOR_SOURCES[item.source]} : ")
        for item in a.transferability.creator_elements
    ] or ["- *Rien d'étayé sur ce point.*"]

    views_per_follower = (
        "non disponible"
        if metrics.views_per_follower is None
        else f"{metrics.views_per_follower:.2f}"
    )
    lines += [
        "",
        "## Contexte de performance",
        "",
        f"**Type de vidéo :** {VIDEO_KINDS[a.performance_context.video_kind]}",
        "",
        f"- Vues par abonné : {views_per_follower}",
        f"- Partages / vues : {_percent(metrics.share_rate)}",
        f"- Sauvegardes / vues : {_percent(metrics.save_rate)}",
        f"- Commentaires / vues : {_percent(metrics.comment_rate)}",
        f"- Likes / vues : {_percent(metrics.like_rate)}",
        "",
        *_claims(a.performance_context.interpretation),
    ]

    removed = len(verification.llm_removals) + len(verification.code_removals)
    corrected = len(verification.llm_corrections) + len(verification.code_corrections)
    lines += [
        "",
        "## Vérification",
        "",
        f"{removed} affirmation(s) retirée(s) faute de preuve, {corrected} correction(s).",
    ]
    lines += [
        f"- Retiré : {item.statement} ({item.reason})"
        for item in [*verification.llm_removals, *verification.code_removals]
    ]
    return "\n".join(lines) + "\n"
