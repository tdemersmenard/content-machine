"""Doubles de test pour les niches."""

from typing import Any

from app.llm.client import ClaudeResult


def draft(**overrides: Any) -> dict[str, Any]:
    return {
        "niche_name": "Entrepreneurs débutants",
        "identity": {
            "who_i_am": "Un gars de Québec qui a lâché sa job",
            "offer": "Du coaching",
            "goal": "leads",
            "goal_details": "10 appels par mois",
        },
        "avatar": {
            "situation": "Salarié de 28 ans tanné de sa job",
            "frustrations": ["J'ai pas le temps", "Je suis écoeuré de mon boss"],
            "desires": ["Être son propre boss"],
            "vocabulary": ["ma job", "tanné"],
            "objections": ["C'est trop risqué"],
            "already_tried": ["Du dropshipping"],
        },
        "dominant_beliefs": ["Il faut un gros capital pour partir"],
        "voice": {
            "tone": "Direct",
            "expressions": ["Mets-en"],
            "never_say": ["synergie"],
            "language": "français",
            "speech_register": "québécois parlé",
        },
        "filming": {
            "on_camera": "yes",
            "equipment": "iPhone",
            "time_available": "2 h par semaine",
            "location": "Chez moi",
            "other_constraints": [],
        },
        "references": {
            "competitors": [{"name": "Concurrent", "why": "Même audience"}],
            "inspirations": [{"name": "keo_fdb", "why": "Parle sans montage"}],
        },
    } | overrides


class FakeNicheClaude:
    """`questions` donne les tours d'entrevue. Le dernier tour (None) termine l'entrevue."""

    def __init__(
        self,
        questions: list[str | None] | None = None,
        fail_on: str | None = None,
        topic_offset: int = 0,
    ) -> None:
        self.questions = list(questions if questions is not None else ["Question 1", None])
        self.fail_on = fail_on
        self.topic_offset = topic_offset
        self.calls: list[dict[str, Any]] = []

    def call_json(self, *, prompt: str, version: str, schema: Any, model: str, **kwargs: Any):
        self.calls.append({"prompt": prompt, "version": version, **kwargs})
        if prompt == self.fail_on:
            raise RuntimeError("Panne simulée")

        if prompt == "niche_interview":
            question = self.questions.pop(0)
            sections = ["identity", "avatar", "beliefs", "voice", "filming", "references"]
            data = schema(
                covered_sections=sections if question is None else sections[:1],
                done=question is None,
                next_question=question or "",
            )
        elif prompt == "niche_profile":
            data = schema.model_validate(draft())
        elif prompt == "niche_topics":
            count = kwargs["variables"]["count"]
            data = schema(
                topics=[
                    {
                        "title": f"Sujet {self.topic_offset + index}",
                        "pain": "Une douleur",
                        "intensity": index % 10 + 1,
                        "rationale": "Parce que",
                        "grounding": "stated" if index % 2 else "inferred",
                    }
                    for index in range(count)
                ]
            )
        else:
            data = schema(
                verbal_tics=["en vrai", "mot jamais dit"],
                signature_expressions=["double your views", "expression inventée"],
                style_summary="Phrases courtes.",
            )
        return ClaudeResult(
            data=data, model=model, prompt_version=f"{prompt}/{version}", attempts=1, cost_usd=0.01
        )

    def count(self, prompt: str) -> int:
        return sum(1 for call in self.calls if call["prompt"] == prompt)
