"""Écriture du profil : un seul chemin, qui versionne et garde tout cohérent."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Niche, NicheVersion
from app.niche.schemas import NicheProfile

ON_CAMERA = {"yes": "Oui", "no": "Non", "mixed": "Parfois", "unknown": "Non renseigné"}
GOALS = {
    "followers": "Abonnés",
    "leads": "Leads",
    "sales": "Ventes",
    "authority": "Autorité",
    "unknown": "Non renseigné",
}
MISSING = "*Non renseigné.*"


def _bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] or [MISSING]


def _text(value: str) -> str:
    return value or MISSING


def render_avatar(name: str, profile: NicheProfile) -> str:
    """Fiche avatar en markdown. Générée depuis le profil, elle ne peut pas le contredire."""
    a = profile.avatar
    lines = [f"# Fiche avatar : {name}", "", "## Qui il est", "", _text(a.situation), ""]
    lines += ["## Ses frustrations", "", *_bullets(a.frustrations), ""]
    lines += ["## Ce qu'il veut", "", *_bullets(a.desires), ""]
    lines += ["## Comment il parle", "", *_bullets(a.vocabulary), ""]
    lines += ["## Ses objections", "", *_bullets(a.objections), ""]
    lines += ["## Ce qu'il a déjà essayé", "", *_bullets(a.already_tried), ""]
    lines += ["## Croyances dominantes de la niche", "", *_bullets(profile.dominant_beliefs), ""]

    i = profile.identity
    lines += [
        "## Moi",
        "",
        f"**Qui je suis :** {_text(i.who_i_am)}",
        "",
        f"**Ce que je vends :** {_text(i.offer)}",
        "",
        f"**Objectif :** {GOALS[i.goal]}. {i.goal_details}".rstrip(),
        "",
    ]

    v = profile.voice
    language = " / ".join(filter(None, (v.language, v.speech_register)))
    lines += [
        "## Ma voix",
        "",
        f"**Ton :** {_text(v.tone)}",
        "",
        f"**Langue et registre :** {_text(language)}",
        "",
        "**Mes expressions**",
        "",
        *_bullets(v.expressions),
        "",
        "**Ce que je ne dirais jamais**",
        "",
        *_bullets(v.never_say),
        "",
    ]
    if profile.voice_profile:
        p = profile.voice_profile
        lines += [
            f"**Mesuré sur {len(p.video_ids)} de mes vidéos :** {p.words_per_second:.1f} mots/s, "
            f"{p.avg_sentence_words:.1f} mots par phrase, "
            f"{p.silences_per_minute:.1f} silences par minute.",
            "",
            p.style_summary,
            "",
            "**Tics de langage**",
            "",
            *_bullets(p.verbal_tics),
            "",
            "**Expressions signature**",
            "",
            *_bullets(p.signature_expressions),
            "",
        ]

    f = profile.filming
    lines += [
        "## Tournage",
        "",
        f"- Face caméra : {ON_CAMERA[f.on_camera]}",
        f"- Matériel : {f.equipment or 'non renseigné'}",
        f"- Temps disponible : {f.time_available or 'non renseigné'}",
        f"- Lieu : {f.location or 'non renseigné'}",
        *[f"- {item}" for item in f.other_constraints],
        "",
    ]

    def people(items: list) -> list[str]:
        return [f"- **{item.name}** : {item.why}".rstrip(" :") for item in items] or [MISSING]

    r = profile.references
    lines += ["## Références", "", "**Concurrents**", "", *people(r.competitors), ""]
    lines += ["**Créateurs de référence**", "", *people(r.inspirations)]
    return "\n".join(lines) + "\n"


def snapshot(session: Session, niche: Niche, change_note: str) -> None:
    """Enregistre une nouvelle version. À appeler après chaque changement."""
    niche.version += 1
    session.add(
        NicheVersion(
            niche_id=niche.id,
            version=niche.version,
            name=niche.name,
            profile=niche.profile,
            topics=niche.topics,
            change_note=change_note,
        )
    )


def set_profile(niche: Niche, profile: NicheProfile) -> None:
    """Le profil fait foi : la fiche et les colonnes historiques en sont dérivées."""
    niche.profile = profile.model_dump(mode="json")
    niche.avatar_markdown = render_avatar(niche.name, profile)
    niche.audience = profile.avatar.situation
    niche.offer = profile.identity.offer
    niche.tone = profile.voice.tone
    niche.banned_words = profile.voice.never_say
    niche.constraints = profile.filming.model_dump(mode="json")


def load_profile(niche: Niche) -> NicheProfile:
    if niche.profile is None:
        raise ValueError(f"La niche {niche.id} n'a pas de profil.")
    return NicheProfile.model_validate(niche.profile)


def deep_merge(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """Fusion récursive des objets. Une liste fournie remplace la liste existante."""
    merged = dict(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def update_niche(
    session: Session,
    niche: Niche,
    changes: dict[str, Any] | None = None,
    name: str | None = None,
    change_note: str = "",
) -> Niche:
    """Applique une édition. Lève une ValidationError si le résultat n'est pas un profil valide."""
    profile = load_profile(niche)
    if changes:
        profile = NicheProfile.model_validate(deep_merge(niche.profile, changes))
    if name is not None:
        niche.name = name
    set_profile(niche, profile)
    snapshot(session, niche, change_note or "Édition manuelle")
    session.commit()
    return niche


def versions(session: Session, niche_id: int) -> list[NicheVersion]:
    return list(
        session.scalars(
            select(NicheVersion)
            .where(NicheVersion.niche_id == niche_id)
            .order_by(NicheVersion.version.desc())
        )
    )
