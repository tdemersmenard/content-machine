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

- **Prompts :** tout prompt vit dans `app/prompts/<nom>_v<N>.md`, jamais en dur dans le code.
  Le fichier contient un frontmatter (`name`, `version`, `description`), une section `# system`
  et une section `# user`. Dans la base, la version s'écrit `<nom>/v<N>`.
- **Versions de prompts :** ne jamais modifier une version déjà utilisée. Créer `<nom>_v<N+1>.md`.
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
  réels portent le marqueur `live`, la vraie transcription le marqueur `whisper`.
- **Doubles de test :** `FakeClaude` et `FakeTranscriber` contournent les vraies bibliothèques.
  Tout nouveau schéma envoyé à Claude doit aussi passer par le vrai SDK dans
  `tests/test_claude_sdk_integration.py`. Après une mise à jour de dépendance, lancer
  `uv run pytest -m whisper`.
- **Dépendance épinglée :** `av<19`, car faster-whisper 1.2.1 appelle un paramètre supprimé
  dans av 19. À retirer quand faster-whisper sera corrigé.
- **Stats des vidéos :** une valeur indisponible reste `null`, jamais estimée ni mise à 0.
- **Étapes d'ingestion :** chaque étape enregistre son résultat et se saute si elle est déjà faite.
  Toute nouvelle étape payante doit respecter cette règle pour qu'une reprise ne repaie rien.
- **Schémas envoyés à Claude :** l'API refuse plus de 24 paramètres optionnels et 16 unions, et une
  grammaire compilée trop grosse (« compiled grammar is too large »). Le schéma d'analyse dépasse
  cette dernière limite : un gros schéma passe `constrained=False`, et son prompt doit contenir
  `{{ output_schema }}`. Pydantic et le retry valident de la même façon dans les deux modes.
  Pas de `| None` ni de valeur par défaut dans les gros schémas : une absence est une liste vide.
- **Preuves de l'analyse :** toute affirmation est un `Claim` avec au moins une preuve horodatée.
  Le contrôle par le code (`app/analysis/evidence.py`) passe après Claude et a le dernier mot.
- **Ratios de performance :** calculés par le code, jamais par Claude.
- **Schéma d'analyse :** il est figé par un snapshot. Pour le changer volontairement :
  `$env:UPDATE_SNAPSHOTS=1; uv run pytest tests/test_analysis_schema.py`, et créer une nouvelle
  version de prompt.
- **Patterns :** un pattern décrit un mécanisme abstrait, sans terme de niche. Les éléments propres
  au créateur (`creator_elements`) ne deviennent jamais des patterns. Un candidat sans preuve
  valable est ignoré. `score_alpha` et `score_beta` ne sont modifiés que par la boucle d'apprentissage.
- **Embeddings :** un vecteur n'est comparable qu'à ceux du même `embedding_model`. Changer de
  modèle impose de recalculer toute la bibliothèque.
- **Profil de niche :** `niches.profile` fait foi. La fiche avatar et les colonnes `audience`, `offer`,
  `tone`, `banned_words` et `constraints` en sont dérivées par `set_profile`, jamais écrites à la main.
  Tout changement passe par `snapshot` pour créer une version.
- **Sujets :** `grounding` distingue ce que le profil dit (`stated`) d'une hypothèse (`inferred`). Le
  générateur doit traiter une hypothèse comme telle.
- **Voix :** seules les vidéos `is_own_content` comptent. Un tic ou une expression absent mot pour
  mot des transcriptions est écarté.
- **Garde-fous des scripts (`app/generate/guards.py`) :** le code refuse plus de 5 mots consécutifs de
  la source ou des exemples de patterns, et les formules `never_say`. Tout chiffre absent du profil
  et des sujets retenus est marqué `[À VÉRIFIER : x]`, sauf les petits nombres sans unité (3 ou moins).
  Les nombres écrits en lettres ne sont pas détectés : le prompt exige les mesures en chiffres.
- **Faits sur le créateur :** un script ne lui invente jamais d'anecdote, de client ou de résultat.
  Il laisse `[À VÉRIFIER : ...]`, et `to_verify` remonte tout en tête du script.
- **Leçons :** une leçon écarte un pattern avec une preuve `{"pattern_id": 3, "effect": "avoid"}` dans
  `evidence`. Le texte de toute leçon active va dans les prompts. `learning` doit respecter ce format.
- **`predicted_score` :** formule non calibrée (60 % note du hook, 40 % moyenne Beta des patterns),
  à recalibrer avec `learning`.
- **Noms de champs Pydantic :** éviter ceux de `BaseModel`, comme `register`. Ils deviennent
  silencieusement optionnels.
- **Timeline :** `timeline.json` est le contrat avec le module d'analyse. Incrémenter
  `TIMELINE_VERSION` à chaque changement de format.

## Commandes

```powershell
uv sync                                                # Installer les dépendances
uv run alembic upgrade head                            # Appliquer les migrations
uv run uvicorn app.main:app --reload                   # Lancer l'API (http://127.0.0.1:8000)
uv run pytest                                          # Tests (sans réseau)
uv run pytest -m live                                  # Test réel de l'API Claude (payant)
uv run pytest -m whisper                               # Vraie transcription (modèle tiny)
uv run pytest -m embeddings                            # Vrai modèle d'embeddings
uv run cm ingest <fichier_ou_url> [--own] [--niche X] [--no-wait]    # Ingérer une vidéo
uv run cm ingest-folder <dossier> [--own] [--niche X] [--no-wait]    # Ingérer un lot
uv run cm status <id>                                  # Statut d'une vidéo
uv run cm analyze <id> [--force]                       # Analyser une vidéo ingérée
uv run cm analyze --all-pending                        # Analyser tout ce qui est en attente
uv run cm patterns list [--type hook] [--niche-id 1]   # Lister les patterns par score
uv run cm patterns show <id>                           # Détail d'un pattern
uv run cm niche new [--name X] [--resume <id>]         # Entrevue d'onboarding (/fin pour finir)
uv run cm niche list                                   # Lister les niches
uv run cm niche show <id>                              # Fiche avatar
uv run cm niche topics <id> [--refresh] [--limit 50]   # Banque de sujets
uv run cm niche voice <id> [--force]                   # Voix depuis mes vidéos
uv run cm adapt <video_id> --niche <id> [-n 3]         # Adapter une vidéo analysée
uv run cm ideas --niche <id> [-n 5] [--develop 3]      # n idées, les meilleures en scripts
uv run cm worker                                       # Traiter la file sans lancer l'API
uv run ruff format .; uv run ruff check .              # Formatage et lint
uv run alembic revision --autogenerate -m "message"    # Créer une migration
uv run alembic check                                   # Vérifier modèles et migrations
uv run alembic downgrade -1                            # Annuler la dernière migration
```

API d'ingestion : `POST /videos` (multipart : `file` ou `url`, plus `is_own_content` et
`niche_source`) et `GET /videos/{id}` (statut, étape, erreur, bundle une fois terminé).
L'API lance un worker en tâche de fond, sauf si `INGEST_WORKER_ENABLED=false`.

API d'analyse : `POST /videos/{id}/analyze` (synchrone, `?force=true` pour refaire) et
`GET /analyses/{id}` (JSON et rapport markdown).

API des patterns : `GET /patterns` (`type`, `niche_id`, `limit`), `GET /patterns/{id}` et
`GET /patterns/search?q=` (`type`, `k`).

API des niches : `POST /niches/interviews`, `POST /niches/interviews/{id}/answers`,
`POST /niches/interviews/{id}/finish`, `GET /niches/interviews/{id}`, `GET /niches`,
`GET /niches/{id}`, `PATCH /niches/{id}` (objets fusionnés, listes remplacées),
`GET /niches/{id}/versions`, `POST /niches/{id}/topics`, `PATCH /niches/{id}/topics/{topic_id}`,
`POST /niches/{id}/voice`.

API des scripts (synchrone, plusieurs minutes) : `POST /scripts/adapt`, `POST /scripts/generate`,
`GET /scripts` (`niche_id`, `status`, `limit`), `PATCH /scripts/{id}` (`status`, `content`,
`published_video_id`). Les scripts sont aussi écrits dans `data/scripts/script_<id>.md`.

Variables d'environnement (`.env`, voir `.env.example`) : `ANTHROPIC_API_KEY`, `MODEL_ANALYSIS`,
`MODEL_GENERATION`, `DATA_DIR`, `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`,
`SCENE_THRESHOLD`, `VISION_BATCH_SIZE`, `MAX_VIDEO_DURATION_S`, `INGEST_WORKER_ENABLED`,
`EMBEDDING_MODEL`, `PATTERN_MERGE_THRESHOLD`.

Prérequis système : ffmpeg et ffprobe dans le PATH.

## État du projet

À mettre à jour à la fin de chaque prompt.

**Dernière mise à jour : 2026-11-05, module de génération.**

Fait :
- Fondations : arborescence, configuration, modèles SQLAlchemy, migration `0001`, wrapper Claude
  (streaming, JSON strict, un retry si invalide, coût, journalisation, fallback sur refus),
  prompts versionnés, endpoint `/health`.
- `app/ingest/` : `ingest(source)`, acquisition (fichier ou yt-dlp), transcription au mot, plans et
  frames clés, lecture des frames par Claude, signaux de rythme, `timeline.json`.
- File de jobs en base (`ingest_jobs`), idempotence par hash, reprise étape par étape.
- `POST /videos`, `GET /videos/{id}`, CLI `cm`. Migration `0002` (`frames`, `ingest_jobs`,
  colonnes de `videos`).
- `app/analysis/` : `analyze(session, video_id)`, passe 1 (`analysis/v1`), passe 2
  (`analysis_verify/v1`), contrôle des preuves par le code, ratios, rapport markdown,
  `POST /videos/{id}/analyze`, `GET /analyses/{id}`, `cm analyze`. Migration `0003`
  (`creator_followers`).
- `app/patterns/` : `extract_patterns`, branché à la fin de `analyze()`, embeddings locaux,
  dédoublonnage par similarité puis décision de Claude, `search_patterns`, `top_patterns`,
  `winning_combinations`, `GET /patterns*`, `cm patterns`. Migration `0004`.
- `app/niche/` : entrevue reprenable, profil structuré, fiche avatar, versions, banque de 50 sujets,
  voix extraite automatiquement après l'ingestion d'une vidéo personnelle, API et `cm niche`.
  Migration `0005`.
- `app/generate/` : `adapt` (squelette, transposition, écriture) et `generate` (Thompson sampling,
  idées, scripts), critique par un viewer impatient avec 2 réécritures au plus, garde-fous, rendu
  markdown, API et `cm adapt`, `cm ideas`. Migration `0006`.

À surveiller :
- `PATTERN_MERGE_THRESHOLD=0.70`, abaissé depuis 0,85 après mesure sur 6 fiches de test : une vraie
  reformulation française du même pattern donne 0,727, une traduction anglaise 0,897, et la paire
  différente la plus proche 0,646. Claude filtre ensuite les faux positifs. À revoir sur de vraies
  données : le seuil n'a été mesuré que sur cet échantillon, et 0,727 est proche de la limite.
- `top_patterns` : tous les scores valent 0,5 tant que `learning` n'existe pas, le classement se
  fait donc sur le nombre d'exemples. Le score reste global, `niche_id` ne fait que filtrer.
- `winning_combinations` n'a pas d'endpoint ni de commande, seulement la fonction. Elle pondère par
  le percentile de vues brutes, qui avantage les gros comptes.

Validé en conditions réelles, sur une seule vidéo (TikTok français de 122 s, plan unique) :
- Toute la chaîne : yt-dlp, Whisper `small` en français, vision, promesse, analyse en 2 passes,
  extraction et fusion de patterns, sans aucun retry. 8 patterns abstraits créés.
- Durée : ingestion 2 min, analyse et patterns 7 min. Coût total 1,68 $ : ingestion 0,26 $, passe 1
  0,62 $, passe 2 0,62 $, patterns 0,19 $.
- La fusion à 0,70 a soumis une paire à Claude (similarité 0,721), qui a refusé à raison : les deux
  fiches étaient bien différentes.

Validé en conditions réelles pour la génération, sur la vraie niche 1 :
- `cm adapt 1 --niche 1 -n 1` : 1,34 $, 7 min 20. `cm ideas --niche 1 -n 5 --develop 1` : environ
  1 $, 5 min 30. Une adaptation à 3 variantes (le défaut) reviendra à environ 3,50 $ et 15 à 20 min.
- Le hook plafonne à 7/10 sur les trois scripts après 2 réécritures. Le critique réclame surtout le
  « comment » : `creator_facts` est le principal levier de qualité. Il manque encore la plateforme,
  le ciblage et le message des pubs, le budget de départ et l'histoire du premier client.
- `script_3` est le premier écrit avec `creator_facts`. `script_1` et `script_2` sont périmés : ils
  enseignent une méthode (porte-à-porte, voisins) qui n'est pas celle du créateur.
- `creator_facts` n'est pas demandé par l'entrevue : il se remplit par `PATCH /niches/{id}`.

Validé en conditions réelles pour la niche, avec un créateur fictif sur une base temporaire :
- Entrevue, profil et banque de sujets en québécois. La voix sur la transcription de keo_fdb.
  0,45 $ pour une entrevue de 14 réponses, environ 6 s d'attente par question.

Non vérifié en conditions réelles :
- Voix sur de vraies vidéos personnelles, via le déclenchement automatique après l'ingestion. Sans
  `voice_profile`, les scripts ne sont pas calés sur la vraie façon de parler du créateur, et la
  durée vient du débit de la vidéo source (`adapt`) ou d'un défaut de 3 mots/s (`generate`).
- Thompson sampling sur de vraies données : tous les scores valent 0,5 et la bibliothèque compte
  8 patterns tirés d'une seule vidéo, donc le tirage est encore presque aléatoire.
- Le critique et l'auteur sont le même modèle : sa note n'est pas une mesure indépendante.
- Vidéos avec montage, texte à l'écran ou cuts : la seule vidéo testée n'en a aucun. Ce n'est pas
  prioritaire : le créateur de référence (keo_fdb) et le format visé sont sans montage.
- Instagram et YouTube : seul TikTok a été essayé.
- Valeur de la passe 2 : elle n'a rien corrigé ni retiré sur cette vidéo, le contrôle par le code non
  plus. Elle coûte autant que la passe 1 car elle réécrit toute l'analyse. À réévaluer sur plus de
  vidéos, par exemple en ne lui faisant renvoyer que les changements.
- Fusion réellement acceptée par Claude entre deux vidéos : pas encore observée.

Pas encore fait :
- Modules `learning`, `agent` et `web/`.
- `POST /videos/{id}/analyze` et `POST /scripts/*` sont synchrones et prennent plusieurs minutes. À
  passer en tâche de fond pour le dashboard.

Décisions :
- Ingestion de vidéos de tiers via yt-dlp validée pour un usage local. À réexaminer si l'outil
  est hébergé ou commercialisé.

- Embeddings locaux avec sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`).

Décisions à prendre :
- Stack du dashboard `web/`.
