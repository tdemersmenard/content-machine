"""Doubles de test pour l'ingestion : aucun réseau, aucun modèle."""

import subprocess
from pathlib import Path
from typing import Any

from app.ingest.acquire import DownloadedVideo
from app.ingest.pipeline import IngestDeps
from app.ingest.schemas import Transcription, Word
from app.llm.client import ClaudeResult

SPOKEN = "Today I will show you how to double your views"


def make_video(path: Path, color: str = "red", voice: bool = True, seconds: int = 9) -> Path:
    """Trois plans de couleur (rouge imposé par `color`, puis bleu et vert), voix synthétique."""
    third = seconds / 3
    command = ["ffmpeg", "-y", "-v", "error"]
    for hue in (color, "blue", "green"):
        command += ["-f", "lavfi", "-i", f"color=c={hue}:s=360x640:d={third}:r=25"]
    graph = "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]"
    maps = ["-map", "[v]"]
    if voice:
        command += ["-f", "lavfi", "-i", f"flite=text='{SPOKEN}':voice=slt"]
        graph += ";[3:a]apad[a]"
        maps += ["-map", "[a]"]
    command += ["-filter_complex", graph, *maps, "-t", str(seconds)]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)]
    subprocess.run(command, check=True, capture_output=True)
    return path


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, audio: Path) -> Transcription:
        assert audio.is_file()
        self.calls += 1
        words = [
            Word(word=word, start=0.5 + index * 0.3, end=0.75 + index * 0.3)
            for index, word in enumerate(SPOKEN.split())
        ]
        return Transcription(words=words, full_text=SPOKEN, language="en")


class FakeClaude:
    """Répond aux prompts `frames` et `promise`. `fail_after` simule une panne en cours de route."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_after = fail_after

    def call_json(self, *, prompt: str, version: str, schema: Any, model: str, **kwargs: Any):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("Panne simulée de Claude")
        self.calls.append({"prompt": prompt, "version": version, "model": model, **kwargs})

        if prompt == "frames":
            data = schema(
                observations=[
                    {
                        "image_number": number,
                        "on_screen_text": "DOUBLE TES VUES",
                        "visual_description": "Un aplat de couleur",
                        "framing": "graphic",
                        "face_present": False,
                        "camera_motion": "static",
                        "graphic_elements": [],
                    }
                    for number in range(1, len(kwargs["images"]) + 1)
                ]
            )
        else:
            data = schema(found=True, promise="double your views", start_word=7, end_word=9)
        return ClaudeResult(
            data=data, model=model, prompt_version=f"{prompt}/{version}", attempts=1, cost_usd=0.01
        )

    def count(self, prompt: str) -> int:
        return sum(1 for call in self.calls if call["prompt"] == prompt)


def make_deps(
    claude: FakeClaude | None = None,
    transcriber: FakeTranscriber | None = None,
    downloads: dict[str, Path] | None = None,
    metadata: dict[str, Any] | None = None,
) -> IngestDeps:
    fake_claude = claude or FakeClaude()

    def download(url: str, destination: Path) -> DownloadedVideo:
        target = destination / "source.mp4"
        target.write_bytes(downloads[url].read_bytes())
        return DownloadedVideo(path=target, metadata=metadata or {})

    return IngestDeps(
        transcriber=transcriber or FakeTranscriber(),
        claude=lambda: fake_claude,
        downloader=download,
    )
