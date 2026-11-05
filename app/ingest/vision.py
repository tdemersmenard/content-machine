"""Lecture des frames par Claude vision, par lots."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from app.llm.client import ClaudeClient

PROMPT_NAME = "frames"
PROMPT_VERSION = "v1"


class FrameObservation(BaseModel):
    image_number: int
    on_screen_text: str
    visual_description: str
    framing: Literal[
        "close_up", "medium_shot", "wide_shot", "b_roll", "screen_record", "graphic", "other"
    ]
    face_present: bool
    camera_motion: Literal["static", "handheld", "pan_tilt", "zoom", "tracking", "unknown"]
    graphic_elements: list[str]


class FrameBatch(BaseModel):
    observations: list[FrameObservation]


def _batch_schema(expected: int) -> type[FrameBatch]:
    """Schéma exigeant une observation par image, pour que le retry du wrapper corrige l'écart."""

    class ExactFrameBatch(FrameBatch):
        @model_validator(mode="after")
        def _one_observation_per_image(self) -> "ExactFrameBatch":
            numbers = sorted(item.image_number for item in self.observations)
            if numbers != list(range(1, expected + 1)):
                raise ValueError(
                    f"Il faut exactement une observation par image, de 1 à {expected} : "
                    f"reçu {numbers}"
                )
            return self

    return ExactFrameBatch


def analyze_batch(
    claude: ClaudeClient, images: list[tuple[float, Path]], model: str
) -> list[FrameObservation]:
    """Analyse un lot de (timestamp, image). Renvoie les observations dans le même ordre."""
    result = claude.call_json(
        prompt=PROMPT_NAME,
        version=PROMPT_VERSION,
        schema=_batch_schema(len(images)),
        model=model,
        variables={"timestamps": [t_s for t_s, _ in images]},
        images=[path for _, path in images],
        effort="medium",
    )
    return sorted(result.data.observations, key=lambda item: item.image_number)
