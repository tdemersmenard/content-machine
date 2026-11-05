# ContentMachine

Analyse des vidéos courtes virales, extraction de patterns, génération de scripts adaptés à une
niche, et apprentissage à partir des stats réelles.

## Démarrage

```powershell
uv sync
Copy-Item .env.example .env    # puis renseigner ANTHROPIC_API_KEY
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

L'API répond sur <http://127.0.0.1:8000/health>. Les tests se lancent avec `uv run pytest`.

La vision, les conventions et l'état du projet sont dans [CLAUDE.md](CLAUDE.md).
