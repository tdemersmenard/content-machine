from pathlib import Path

import pytest

from app.llm.errors import PromptError
from app.llm.prompts import load_prompt


def write_prompt(root: Path, body: str, name: str = "demo", version: str = "v1") -> None:
    (root / f"{name}_{version}.md").write_bytes(body.encode("utf-8"))


VALID = "---\nname: demo\nversion: v1\n---\n# system\nStatique.\n\n# user\nBonjour {{ nom }}\n"


def test_load_shipped_prompt() -> None:
    prompt = load_prompt("smoke", "v1")

    assert prompt.prompt_version == "smoke/v1"
    assert "mot" in prompt.render_user({"word": "mot"})


def test_sections_are_parsed_and_rendered(tmp_path: Path) -> None:
    write_prompt(tmp_path, VALID)

    prompt = load_prompt("demo", "v1", tmp_path)

    assert prompt.system == "Statique."
    assert prompt.render_user({"nom": "Ada"}) == "Bonjour Ada"


def test_missing_variable_raises(tmp_path: Path) -> None:
    write_prompt(tmp_path, VALID)

    with pytest.raises(PromptError):
        load_prompt("demo", "v1", tmp_path).render_user({})


def test_hash_ignores_line_endings_but_detects_edits(tmp_path: Path) -> None:
    write_prompt(tmp_path, VALID)
    path = tmp_path / "demo_v1.md"
    lf = load_prompt("demo", "v1", tmp_path)

    path.write_bytes(VALID.replace("\n", "\r\n").encode("utf-8"))
    crlf = load_prompt("demo", "v1", tmp_path)

    path.write_bytes(VALID.replace("Statique", "Modifié").encode("utf-8"))
    edited = load_prompt("demo", "v1", tmp_path)

    assert crlf.content_hash == lf.content_hash
    assert edited.content_hash != lf.content_hash


@pytest.mark.parametrize(
    "body",
    [
        "# system\nsans frontmatter\n# user\nx\n",
        "---\nname: demo\nversion: v1\n---\n# user\nsans system\n",
        "---\nname: autre\nversion: v1\n---\n# system\na\n# user\nb\n",
        "---\nname: demo\nversion: v1\n---\n# system\n{{ dynamique }}\n# user\nb\n",
    ],
)
def test_malformed_prompts_are_rejected(tmp_path: Path, body: str) -> None:
    write_prompt(tmp_path, body)

    with pytest.raises(PromptError):
        load_prompt("demo", "v1", tmp_path)


@pytest.mark.parametrize(("name", "version"), [("../secret", "v1"), ("smoke", "latest")])
def test_unsafe_identifiers_are_rejected(name: str, version: str) -> None:
    with pytest.raises(PromptError):
        load_prompt(name, version)


def test_missing_prompt_raises() -> None:
    with pytest.raises(PromptError, match="introuvable"):
        load_prompt("inexistant", "v1")
