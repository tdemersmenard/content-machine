"""Configuration de l'application, lue depuis l'environnement et le fichier .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL = "claude-opus-5-5"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    model_analysis: str = DEFAULT_MODEL
    model_generation: str = DEFAULT_MODEL
    data_dir: Path = Path("data")

    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    scene_threshold: float = 0.3
    vision_batch_size: int = 10
    # Garde-fou de coût : le nombre de frames envoyées à Claude croît avec la durée.
    max_video_duration_s: float = 600.0
    ingest_worker_enabled: bool = True

    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    # Mesuré : une reformulation française du même pattern donne environ 0,73.
    pattern_merge_threshold: float = 0.70

    learning_half_life_days: float = 90.0  # Un résultat pèse moitié moins après cette durée.
    learning_min_baseline: int = 3  # Relevés précédents requis pour calculer un score.
    learning_exploration_rate: float = 0.2

    # Au-delà, une idée est considérée comme déjà produite. Voir CLAUDE.md pour la mesure.
    idea_similarity_threshold: float = 0.60

    model_agent: str = DEFAULT_MODEL
    agent_scheduler_enabled: bool = True
    agent_timezone: str = "America/Toronto"
    # Tokens facturés par jour (entrée, sortie, écriture de cache). Environ 9 $ au tarif Opus 5.5.
    agent_daily_token_budget: int = 1_000_000
    agent_max_tool_rounds: int = 15
    agent_daily_hour: int = 6
    agent_weekly_hour: int = 7
    agent_weekly_scripts: int = 5  # 0 désactive le plan de contenu de la semaine.
    agent_inbox_interval_s: int = 60

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def database_url(self) -> str:
        return f"sqlite:///{(self.data_dir / 'contentmachine.db').resolve().as_posix()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
