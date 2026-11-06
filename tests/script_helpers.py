"""Doubles de test pour la génération de scripts."""

from typing import Any

from app.llm.client import ClaudeResult
from app.llm.errors import ClaudeInvalidOutputError


def script_draft(pattern_ids: list[int], **overrides: Any) -> dict[str, Any]:
    return {
        "internal_title": "Le vrai problème",
        "angle": "Tu cherches au mauvais endroit",
        "belief_attacked": "Il faut être né vendeur",
        "hooks": [
            {
                "spoken": f"Hook numéro {letter} qui accroche ben raide",
                "on_screen_text": "",
                "visual": "Face caméra",
                "hook_type": "contrarian",
            }
            for letter in "abc"
        ],
        "beats": [
            {
                "role": "setup",
                "target_start_s": 0,
                "target_end_s": 0,
                "spoken": "Un deux trois quatre cinq six",
                "on_screen_text": "",
                "visual": "Face caméra",
                "performance_note": "Pause",
            },
            {
                "role": "payoff",
                "target_start_s": 0,
                "target_end_s": 0,
                "spoken": "Sept huit neuf",
                "on_screen_text": "",
                "visual": "Face caméra",
                "performance_note": "Regard",
            },
        ],
        "open_loops": [{"description": "Quoi faire", "opened_beat": 0, "closed_beat": 1}],
        "ending": "Essaie ça",
        "cta": "Abonne-toi",
        "caption": "La vérité",
        "hashtags": ["business"],
        "pattern_ids": pattern_ids,
        "rationale": [{"choice": "Hook", "reason": "Il claque", "source": "profile"}],
    } | overrides


class FakeScriptClaude:
    """`scores` donne les notes successives du critique, `drafts` les textes écrits."""

    def __init__(
        self,
        scores: list[int] | None = None,
        drafts: list[dict] | None = None,
        fail_on: str | None = None,
        fail_write_at: int | None = None,
    ) -> None:
        self.scores = list(scores if scores is not None else [9] * 50)
        self.drafts = list(drafts or [])
        self.fail_on = fail_on
        self.fail_write_at = fail_write_at
        self.calls: list[dict[str, Any]] = []

    def call_json(self, *, prompt: str, version: str, schema: Any, model: str, **kwargs: Any):
        self.calls.append({"prompt": prompt, "version": version, "schema": schema, **kwargs})
        if prompt == self.fail_on:
            raise RuntimeError("Panne simulée")
        return ClaudeResult(
            data=self._reply(prompt, schema, kwargs["variables"]),
            model=model,
            prompt_version=f"{prompt}/{version}",
            attempts=1,
            cost_usd=0.01,
        )

    def _reply(self, prompt: str, schema: Any, variables: dict[str, Any]):
        if prompt == "script_skeleton":
            return schema.model_validate(
                {
                    "beats": [
                        {
                            "role": "setup",
                            "mechanism": "Installe une tension abstraite",
                            "start_pct": 0,
                            "end_pct": 50,
                        },
                        {
                            "role": "payoff",
                            "mechanism": "Résout la tension",
                            "start_pct": 50,
                            "end_pct": 100,
                        },
                    ]
                }
            )
        if prompt == "script_transpose":
            return schema.model_validate(
                {
                    "variants": [
                        {
                            "core_idea": f"Idée de fond {index}",
                            "angle": f"Angle {index}",
                            "belief_attacked": "Il faut un gros capital",
                            "topic_ids": [],
                            "beats": [
                                {
                                    "beat_index": beat,
                                    "pain": "Douleur",
                                    "belief": "",
                                    "proof": "Preuve",
                                    "example": "",
                                }
                                for beat in range(2)
                            ],
                        }
                        for index in range(variables["count"])
                    ]
                }
            )
        if prompt == "script_ideas":
            import json

            topics = json.loads(variables["topics"])
            return schema.model_validate(
                {
                    "ideas": [
                        {
                            "combination_index": index,
                            "topic_id": topics[index]["id"],
                            "core_idea": f"Idée de fond {index}",
                            "angle": f"Idée {index}",
                            "belief_attacked": "Croyance",
                            "why": "Parce que",
                            "potential": index + 1,
                        }
                        for index in range(variables["count"])
                    ]
                }
            )
        if prompt in ("script_write", "script_rewrite"):
            if self.fail_write_at is not None and self.count("script_write") > self.fail_write_at:
                raise RuntimeError("Panne simulée")
            if not self.drafts:
                raise ClaudeInvalidOutputError("Plus de brouillon de test", "")
            return schema.model_validate(self.drafts.pop(0))
        return schema(
            hook_score=self.scores.pop(0),
            swipe_points=[
                {"quote": "Un deux trois", "reason": "Trop long"},
                {"quote": "phrase absente du script", "reason": "Inventée"},
            ],
            fixes=["Va droit au but"],
        )

    def count(self, prompt: str) -> int:
        return sum(1 for call in self.calls if call["prompt"] == prompt)
