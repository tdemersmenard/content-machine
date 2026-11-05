# ContentMachine

## Vision

ContentMachine analyse des vidéos courtes virales (TikTok, Reels, Shorts), comprend pourquoi elles
fonctionnent et en extrait des patterns réutilisables. Il génère ensuite des scripts adaptés à la
niche de l'utilisateur, puis apprend des stats réelles des vidéos publiées pour s'améliorer en
continu.

Boucle produit : ingestion → analyse → patterns → génération → publication → mesure → leçons.

## Structure

| Dossier | Rôle |
|---|---|
| `app/ingest/` | Téléchargement, transcription, frames, métadonnées |
| `app/analysis/` | Moteur d'analyse virale |
| `app/patterns/` | Bibliothèque de patterns et embeddings |
| `app/niche/` | Profils de niche |
| `app/generate/` | Génération de scripts |
| `app/learning/` | Stats, scoring, leçons |
| `app/agent/` | Orchestrateur autonome |
| `app/api/` | Routes FastAPI |
| `app/db/` | Base déclarative, session, modèles SQLAlchemy |
| `app/llm/` | Wrapper Claude, chargement des prompts, tarifs |
| `app/prompts/` | Tous les prompts envoyés à Claude |
| `web/` | Dashboard |
| `data/` | Vidéos, frames, exports, base SQLite (gitignore) |
| `tests/` | Tests pytest |

## Conventions

- **Prompts :** tout prompt vit dans `app/prompts/<nom>/v<N>.md`, jamais en dur dans le code.
  Le fichier contient un frontmatter (`name`, `version`, `description`), une section `# system`
  et une section `# user`.
- **Versions de prompts :** ne jamais modifier une version déjà utilisée. Créer `v<N+1>.md`.
  La colonne `prompt_hash` de `llm_calls` révèle les modifications faites sans changer de version.
- **System prompt statique :** les variables (Jinja2) vont uniquement dans `# user`, pour que le
  prompt caching fonctionne. Le loader refuse un system prompt contenant une variable.
- **Sorties de Claude :** toujours du JSON validé par un modèle Pydantic. Tout appel passe par
  `ClaudeClient.call_json` (`get_claude()` dans `app/llm/client.py`), jamais par le SDK directement.
- **Journalisation :** chaque tentative d'appel écrit une ligne dans `llm_calls` avec
  `prompt_version`, les tokens et le coût. Quand un résultat est stocké (`analyses`, `scripts`),
  y copier `prompt_version` et le modèle depuis `ClaudeResult`.
- **Paramètres API :** pas de `temperature`, `top_p` ni `top_k` (rejetés par l'API). La profondeur
  de raisonnement se règle avec `effort`. Pas de `budget_tokens` ni de prefill.
- **Nouveau modèle Claude :** ajouter son tarif dans `app/llm/pricing.py`, sinon `cost_usd` reste
  `NULL`. Ajouter aussi le modèle à `FALLBACK_MODELS` s'il supporte le fallback.
- **Base de données :** toute modification de schéma passe par une migration Alembic. Les datetimes
  doivent avoir un fuseau horaire (`utcnow()` de `app/db/base.py`), un datetime naïf est refusé.
- **Tests :** aucun test par défaut n'appelle le réseau ni ne nécessite de clé API. Les appels
  réels portent le marqueur `live`.

## Commandes

```powershell
uv sync                                                # Installer les dépendances
uv run alembic upgrade head                            # Appliquer les migrations
uv run uvicorn app.main:app --reload                   # Lancer l'API (http://127.0.0.1:8000)
uv run pytest                                          # Tests (sans réseau)
uv run pytest -m live                                  # Test réel de l'API Claude (payant)
uv run ruff format .; uv run ruff check .              # Formatage et lint
uv run alembic revision --autogenerate -m "message"    # Créer une migration
uv run alembic check                                   # Vérifier modèles et migrations
uv run alembic downgrade -1                            # Annuler la dernière migration
```

Variables d'environnement (`.env`, voir `.env.example`) : `ANTHROPIC_API_KEY`, `MODEL_ANALYSIS`,
`MODEL_GENERATION`, `DATA_DIR`.

## État du projet

À mettre à jour à la fin de chaque prompt.

**Dernière mise à jour : 2026-11-05, prompt 1 (fondations).**

Fait :
- Arborescence, configuration, modèles SQLAlchemy et migration `0001` (10 tables + `llm_calls`).
- Wrapper Claude : streaming, sortie JSON stricte, un retry si invalide, coût, journalisation,
  fallback serveur sur refus.
- Chargement des prompts versionnés et endpoint `/health`.

Pas encore fait :
- Modules métier (`ingest`, `analysis`, `patterns`, `niche`, `generate`, `learning`, `agent`) et `web/`.
- Test `live` jamais exécuté contre la vraie API : aucune clé n'était disponible.

Décisions à prendre :
- Fournisseur d'embeddings (Voyage ou modèle local). Anthropic n'a pas d'API d'embeddings.
- Périmètre de l'ingestion au regard des CGU de TikTok, Instagram et YouTube.
- Stack du dashboard `web/`.

Prérequis pour la suite : ffmpeg n'est pas installé (nécessaire à l'extraction des frames).
