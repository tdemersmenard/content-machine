"""Chargement des prompts versionnés de app/prompts/<nom>_v<N>.md.

Format d'un fichier :

    ---
    name: smoke
    version: v1
    description: À quoi sert ce prompt
    ---
    # system
    Texte statique, sans variable, pour rester cacheable.
    # user
    Texte Jinja2 : {{ variable }}
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined, TemplateError

from app.llm.errors import PromptError

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_NAME_RE = re.compile(r"^[a-z0-9_]+$")
_VERSION_RE = re.compile(r"^v\d+$")
_FILE_RE = re.compile(
    r"\A---\n(?P<meta>.*?)\n---\n+# system\n(?P<system>.*?)\n# user\n(?P<user>.*)\Z", re.DOTALL
)
_jinja = Environment(undefined=StrictUndefined, autoescape=False)


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    system: str
    user_template: str
    content_hash: str

    @property
    def prompt_version(self) -> str:
        return f"{self.name}/{self.version}"

    def render_user(self, variables: dict[str, Any] | None = None) -> str:
        try:
            return _jinja.from_string(self.user_template).render(variables or {})
        except TemplateError as exc:
            raise PromptError(f"{self.prompt_version} : {exc}") from exc


def load_prompt(name: str, version: str, prompts_dir: Path = PROMPTS_DIR) -> Prompt:
    if not _NAME_RE.match(name) or not _VERSION_RE.match(version):
        raise PromptError(f"Nom ou version de prompt invalide : {name!r}, {version!r}")

    path = prompts_dir / f"{name}_{version}.md"
    if not path.is_file():
        raise PromptError(f"Prompt introuvable : {path}")

    # Fins de ligne normalisées : le hash reste le même sous Windows et Linux.
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    match = _FILE_RE.match(text)
    if match is None:
        raise PromptError(f"{path} : frontmatter, '# system' et '# user' sont obligatoires.")

    meta = yaml.safe_load(match["meta"]) or {}
    if meta.get("name") != name or meta.get("version") != version:
        raise PromptError(f"{path} : le frontmatter ne correspond pas au chemin du fichier.")

    system = match["system"].strip()
    if "{{" in system or "{%" in system:
        raise PromptError(f"{path} : le system prompt doit rester statique (prompt caching).")

    return Prompt(
        name=name,
        version=version,
        system=system,
        user_template=match["user"].strip(),
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    )
