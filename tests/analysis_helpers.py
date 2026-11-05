"""Données et doubles de test pour l'analyse."""

from typing import Any

from app.llm.client import ClaudeResult


def quote(text: str = "double your views", start: float = 2.6, end: float = 3.45) -> dict:
    return {"kind": "quote", "start_s": start, "end_s": end, "excerpt": text}


def claim(statement: str = "Une affirmation", *evidence: dict) -> dict:
    return {"statement": statement, "evidence": list(evidence) or [quote()]}


def sample_analysis() -> dict[str, Any]:
    """Une analyse complète, dont chaque preuve existe dans la vidéo de test."""
    screen = {"kind": "on_screen_text", "start_s": 0, "end_s": 0, "excerpt": "DOUBLE TES VUES"}
    frame = {"kind": "frame", "start_s": 1, "end_s": 1, "excerpt": "Un aplat de couleur rouge"}
    signal = {"kind": "signal", "start_s": 0, "end_s": 9, "excerpt": "shot_count = 3"}
    return {
        "hook": {
            "spoken_text": [claim("Il annonce le résultat")],
            "on_screen_text": [claim("Le texte promet le résultat", screen)],
            "visual": [claim("Fond uni", frame)],
            "hook_type": "result_first",
            "hook_type_other": "",
            "hook_type_justification": [claim("Le résultat vient d'abord")],
            "psychological_mechanism": [claim("Gain immédiat promis")],
            "alignment": "complementary",
            "alignment_explanation": [claim("Le texte renforce la voix", screen)],
            "score": 7,
            "score_justification": [claim("Promesse claire")],
        },
        "angle": {
            "raw_topic": [claim("Les vues")],
            "chosen_angle": [claim("Doubler ses vues")],
            "audience_beliefs": [{"stance": "attacks", "claim": claim("Croire que c'est lent")}],
            "implicit_promise": [claim("Tu vas apprendre la méthode")],
            "promise_payoff": [],
            "alternatives": [
                {"angle": f"Angle {n}", "comparison": "original_better", "claim": claim()}
                for n in range(3)
            ],
        },
        "structure": {
            "beats": [
                {"start_s": 0, "end_s": 3, "role": "hook", "claim": claim("Le hook")},
                {"start_s": 3, "end_s": 9, "role": "setup", "claim": claim("La suite", frame)},
            ],
            "framework": "tutorial",
            "framework_justification": [claim("Annonce une méthode")],
        },
        "retention": {
            "open_loops": [{"opened": claim("Comment doubler ?"), "closed": []}],
            "rehooks": [claim("Changement de plan", signal)],
            "drop_risks": [{"risk": claim("Fond uni", frame), "compensation": [claim()]}],
            "rhythm": [claim("Peu de plans", signal)],
            "ending_type": "abrupt",
            "ending_effect": [claim("Fin sans payoff", frame)],
        },
        "engagement": {
            "comments": [{"trigger": "open_question", "claim": claim("Laisse la question")}],
            "shares": [{"trigger": "utility", "claim": claim("Utile")}],
            "saves": [claim("Méthode à garder")],
        },
        "emotion_curve": [
            {
                "start_s": 0,
                "end_s": 3,
                "emotion": "curiosité",
                "intensity": 6,
                "claim": claim("Promesse"),
            }
        ],
        "transferability": {
            "format_elements": [claim("La promesse chiffrée")],
            "creator_elements": [{"source": "other", "claim": claim("Voix", frame)}],
        },
        "verdict": {
            "reasons": [
                {
                    "rank": rank,
                    "weight_percent": 40,
                    "confidence": "low",
                    "claim": claim(f"Raison {rank}"),
                }
                for rank in (1, 2, 3)
            ]
        },
        "performance_context": {"video_kind": "unknown", "interpretation": []},
    }


class FakeAnalysisClaude:
    """Renvoie `draft` à la passe 1 et `verified` (ou le brouillon) à la passe 2."""

    def __init__(
        self,
        draft: dict | None = None,
        verified: dict | None = None,
        corrections: list | None = None,
        removals: list | None = None,
        fail_on: str | None = None,
    ) -> None:
        self.draft = draft or sample_analysis()
        self.verified = verified
        self.corrections = corrections or []
        self.removals = removals or []
        self.fail_on = fail_on
        self.calls: list[dict[str, Any]] = []

    def call_json(self, *, prompt: str, version: str, schema: Any, model: str, **kwargs: Any):
        if prompt == "pattern_extract" and self.fail_on != prompt:
            return ClaudeResult(
                data=schema(candidates=[]),
                model=model,
                prompt_version=f"{prompt}/{version}",
                attempts=1,
                cost_usd=0.0,
            )
        self.calls.append({"prompt": prompt, "version": version, **kwargs})
        if prompt == self.fail_on:
            raise RuntimeError("Panne simulée")

        if prompt == "analysis":
            data = schema.model_validate(self.draft)
        else:
            data = schema.model_validate(
                {
                    "analysis": self.verified or self.draft,
                    "corrections": self.corrections,
                    "removals": self.removals,
                }
            )
        return ClaudeResult(
            data=data, model=model, prompt_version=f"{prompt}/{version}", attempts=1, cost_usd=0.1
        )

    def count(self, prompt: str) -> int:
        return sum(1 for call in self.calls if call["prompt"] == prompt)
