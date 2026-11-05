"""Entrevue, profil, versions, sujets et voix."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from app import cli as cli_module
from app.cli import cli
from app.db.models import InterviewStatus, Niche, NicheInterview, NicheVersion
from app.ingest.jobs import ingest
from app.ingest.pipeline import IngestDeps
from app.llm.client import get_claude
from app.llm.prompts import load_prompt
from app.main import app
from app.niche.interview import (
    MAX_ANSWERS,
    InterviewError,
    answer_interview,
    current_question,
    finish_interview,
    start_interview,
)
from app.niche.profile import deep_merge, load_profile, render_avatar, update_niche, versions
from app.niche.schemas import InterviewTurn, NicheProfile
from app.niche.topics import TARGET_TOPICS, refresh_topics, set_topic_status
from app.niche.voice import (
    contains_phrase,
    refresh_voice,
    repeated_phrases,
    sentence_lengths,
)
from tests.ingest_helpers import FakeClaude, FakeTranscriber, make_deps, make_video
from tests.niche_helpers import FakeNicheClaude, draft


def make_niche(db: Session, claude: FakeNicheClaude | None = None) -> Niche:
    fake = claude or FakeNicheClaude()
    interview = start_interview(db, fake)
    answer_interview(db, interview, "Ma réponse", fake)
    return db.get(Niche, interview.niche_id)


def count(db: Session, model: type) -> int:
    return db.scalar(select(func.count()).select_from(model))


def test_interview_asks_one_question_at_a_time(db: Session) -> None:
    fake = FakeNicheClaude(["Q1", "Q2", None])

    interview = start_interview(db, fake)
    assert current_question(interview) == "Q1"

    answer_interview(db, interview, "R1", fake)
    assert current_question(interview) == "Q2"
    assert interview.status is InterviewStatus.IN_PROGRESS
    assert [item["role"] for item in interview.transcript] == [
        "assistant",
        "user",
        "assistant",
    ]


def test_each_turn_sends_the_whole_conversation(db: Session) -> None:
    fake = FakeNicheClaude(["Q1", "Q2", None])
    interview = start_interview(db, fake)

    answer_interview(db, interview, "Ma réponse précise", fake)

    assert "Ma réponse précise" in fake.calls[1]["variables"]["transcript"]


def test_interview_end_creates_the_niche_profile_and_topics(db: Session) -> None:
    fake = FakeNicheClaude()
    interview = start_interview(db, fake)

    answer_interview(db, interview, "R1", fake)

    assert interview.status is InterviewStatus.DONE
    assert current_question(interview) is None
    niche = db.get(Niche, interview.niche_id)
    assert niche.name == "Entrepreneurs débutants"
    assert load_profile(niche).avatar.vocabulary == ["ma job", "tanné"]
    assert len(niche.topics) == TARGET_TOPICS
    assert "Fiche avatar" in niche.avatar_markdown


def test_derived_columns_follow_the_profile(db: Session) -> None:
    niche = make_niche(db)

    assert niche.audience == "Salarié de 28 ans tanné de sa job"
    assert niche.offer == "Du coaching"
    assert niche.tone == "Direct"
    assert niche.banned_words == ["synergie"]
    assert niche.constraints["equipment"] == "iPhone"


def test_given_name_wins_and_duplicates_get_a_suffix(db: Session) -> None:
    make_niche(db)
    second = make_niche(db)
    fake = FakeNicheClaude()
    named = start_interview(db, fake, name="Ma niche")
    answer_interview(db, named, "R", fake)

    assert second.name == "Entrepreneurs débutants (2)"
    assert db.get(Niche, named.niche_id).name == "Ma niche"


def test_failed_turn_keeps_nothing_so_the_answer_can_be_resent(db: Session) -> None:
    fake = FakeNicheClaude(["Q1", "Q2", None])
    interview = start_interview(db, fake)
    fake.fail_on = "niche_interview"

    with pytest.raises(RuntimeError):
        answer_interview(db, interview, "R1", fake)

    db.expire_all()
    assert len(db.get(NicheInterview, interview.id).transcript) == 1
    assert current_question(interview) == "Q1"

    fake.fail_on = None
    answer_interview(db, interview, "R1", fake)
    assert current_question(interview) == "Q2"


def test_failed_synthesis_creates_no_niche_and_can_be_retried(db: Session) -> None:
    fake = FakeNicheClaude(fail_on="niche_profile")
    interview = start_interview(db, fake)

    with pytest.raises(RuntimeError):
        answer_interview(db, interview, "R1", fake)

    assert count(db, Niche) == 0
    assert db.get(NicheInterview, interview.id).status is InterviewStatus.IN_PROGRESS


def test_topic_failure_keeps_the_paid_profile(db: Session) -> None:
    fake = FakeNicheClaude(fail_on="niche_topics")
    interview = start_interview(db, fake)

    answer_interview(db, interview, "R1", fake)

    niche = db.get(Niche, interview.niche_id)
    assert niche.profile is not None
    assert niche.topics == []
    assert interview.status is InterviewStatus.DONE


def test_early_finish_uses_what_was_said(db: Session) -> None:
    fake = FakeNicheClaude(["Q1", "Q2"])
    interview = start_interview(db, fake)
    answer_interview(db, interview, "R1", fake)

    finish_interview(db, interview, fake)

    assert interview.status is InterviewStatus.DONE
    assert fake.count("niche_profile") == 1


def test_finish_without_any_answer_is_refused(db: Session) -> None:
    fake = FakeNicheClaude(["Q1"])
    interview = start_interview(db, fake)

    with pytest.raises(InterviewError, match="aucune réponse"):
        finish_interview(db, interview, fake)


def test_finishing_twice_creates_one_niche(db: Session) -> None:
    fake = FakeNicheClaude()
    interview = start_interview(db, fake)
    answer_interview(db, interview, "R1", fake)

    finish_interview(db, interview, fake)

    assert count(db, Niche) == 1


def test_answering_a_finished_interview_is_refused(db: Session) -> None:
    fake = FakeNicheClaude()
    interview = start_interview(db, fake)
    answer_interview(db, interview, "R1", fake)

    with pytest.raises(InterviewError, match="terminée"):
        answer_interview(db, interview, "R2", fake)


def test_empty_answer_is_refused(db: Session) -> None:
    fake = FakeNicheClaude(["Q1"])
    interview = start_interview(db, fake)

    with pytest.raises(InterviewError, match="vide"):
        answer_interview(db, interview, "   ", fake)


def test_interview_stops_at_the_answer_limit(db: Session) -> None:
    fake = FakeNicheClaude([f"Q{n}" for n in range(MAX_ANSWERS + 5)])
    interview = start_interview(db, fake)

    for _ in range(MAX_ANSWERS):
        answer_interview(db, interview, "R", fake)

    assert interview.status is InterviewStatus.DONE


def test_interview_cannot_finish_with_missing_sections() -> None:
    with pytest.raises(ValidationError, match="6 sections"):
        InterviewTurn(covered_sections=["identity"], done=True, next_question="")
    with pytest.raises(ValidationError, match="prochaine question"):
        InterviewTurn(covered_sections=[], done=False, next_question=" ")


def test_deep_merge_keeps_siblings_and_replaces_lists() -> None:
    base = {"a": {"x": 1, "y": 2}, "l": [1, 2]}

    assert deep_merge(base, {"a": {"x": 9}, "l": [3]}) == {"a": {"x": 9, "y": 2}, "l": [3]}
    assert base["a"]["x"] == 1


def test_edit_changes_one_field_and_rebuilds_everything_derived(db: Session) -> None:
    niche = make_niche(db)

    update_niche(db, niche, {"voice": {"tone": "Baveux"}, "identity": {"offer": "Formation"}})

    profile = load_profile(niche)
    assert profile.voice.tone == "Baveux"
    assert profile.voice.language == "français"
    assert niche.tone == "Baveux"
    assert niche.offer == "Formation"
    assert "Formation" in niche.avatar_markdown


def test_invalid_edit_changes_nothing(db: Session) -> None:
    niche = make_niche(db)
    version = niche.version

    with pytest.raises(ValidationError):
        update_niche(db, niche, {"identity": {"goal": "gloire"}})

    db.rollback()
    assert niche.version == version
    assert load_profile(niche).identity.goal == "leads"


def test_creator_facts_are_editable_and_shown_on_the_sheet(db: Session) -> None:
    niche = make_niche(db)
    assert load_profile(niche).creator_facts == []

    update_niche(db, niche, {"creator_facts": ["A trouvé ses clients grâce aux pubs"]})

    assert load_profile(niche).creator_facts == ["A trouvé ses clients grâce aux pubs"]
    assert "- A trouvé ses clients grâce aux pubs" in niche.avatar_markdown


def test_profile_stored_before_creator_facts_still_loads(db: Session) -> None:
    niche = make_niche(db)
    niche.profile = {k: v for k, v in niche.profile.items() if k != "creator_facts"}
    db.commit()

    assert load_profile(niche).creator_facts == []


def test_every_change_creates_a_version_with_the_full_state(db: Session) -> None:
    niche = make_niche(db)
    start = niche.version

    update_niche(db, niche, {"voice": {"tone": "A"}}, change_note="Ton plus direct")
    update_niche(db, niche, {"voice": {"tone": "B"}})

    history = versions(db, niche.id)
    assert niche.version == start + 2
    assert [item.version for item in history][:2] == [start + 2, start + 1]
    assert history[0].profile["voice"]["tone"] == "B"
    assert history[1].profile["voice"]["tone"] == "A"
    assert history[1].change_note == "Ton plus direct"
    assert history[-1].version == 1
    assert history[-1].change_note == "Création par entrevue"


def test_rename_is_versioned(db: Session) -> None:
    niche = make_niche(db)

    update_niche(db, niche, name="Nouveau nom")

    assert versions(db, niche.id)[0].name == "Nouveau nom"
    assert "Nouveau nom" in niche.avatar_markdown


def test_avatar_sheet_shows_what_is_missing_without_inventing() -> None:
    data = draft()
    data["avatar"]["objections"] = []
    data["identity"]["goal"] = "unknown"

    sheet = render_avatar("Test", NicheProfile.model_validate(data))

    assert "## Ses objections\n\n*Non renseigné.*" in sheet
    assert "**Objectif :** Non renseigné." in sheet
    assert "**keo_fdb** : Parle sans montage" in sheet


def test_bank_has_50_topics_sorted_by_intensity(db: Session) -> None:
    niche = make_niche(db)

    intensities = [item["intensity"] for item in niche.topics]

    assert len(niche.topics) == 50
    assert intensities == sorted(intensities, reverse=True)
    assert all(item["status"] == "new" and item["id"] for item in niche.topics)
    assert len({item["id"] for item in niche.topics}) == 50


def test_full_bank_is_not_regenerated(db: Session) -> None:
    fake = FakeNicheClaude()
    niche = make_niche(db, fake)
    calls = fake.count("niche_topics")

    refresh_topics(db, niche, fake)

    assert fake.count("niche_topics") == calls


def test_refresh_replaces_used_topics_and_keeps_history(db: Session) -> None:
    niche = make_niche(db)
    used = niche.topics[0]["id"]
    retired = niche.topics[1]["id"]
    set_topic_status(db, niche, used, "used")
    set_topic_status(db, niche, retired, "retired")
    fake = FakeNicheClaude(topic_offset=1000)

    refresh_topics(db, niche, fake)

    assert fake.calls[0]["variables"]["count"] == 2
    assert "Sujet 1" in fake.calls[0]["variables"]["existing"]
    assert sum(item["status"] == "new" for item in niche.topics) == 50
    assert len(niche.topics) == 52
    assert {used, retired} <= {item["id"] for item in niche.topics}


def test_repeated_titles_are_not_added_twice(db: Session) -> None:
    niche = make_niche(db)
    set_topic_status(db, niche, niche.topics[0]["id"], "used")

    refresh_topics(db, niche, FakeNicheClaude())

    assert len({item["title"] for item in niche.topics}) == len(niche.topics)


def test_wrong_topic_count_is_rejected_for_the_wrapper_retry() -> None:
    from app.niche.topics import _batch_schema

    with pytest.raises(ValidationError, match="exactement 2"):
        _batch_schema(2)(topics=[])


def test_unknown_topic_raises(db: Session) -> None:
    niche = make_niche(db)

    with pytest.raises(LookupError):
        set_topic_status(db, niche, "inconnu", "used")


def test_sentence_lengths() -> None:
    assert sentence_lengths("Salut toi. Ça va bien ? Oui !") == [2, 3, 1]
    assert sentence_lengths("") == []


def test_repeated_phrases_need_two_occurrences() -> None:
    phrases = dict(repeated_phrases(["en vrai je pense", "en vrai tu sais"]))

    assert phrases["en vrai"] == 2
    assert "je pense" not in phrases


def test_contains_phrase_ignores_case_and_punctuation() -> None:
    assert contains_phrase(["Tu sais, EN VRAI !"], "en vrai")
    assert not contains_phrase(["en fait vrai"], "en vrai")
    assert not contains_phrase(["texte"], "...")


def own_video(db: Session, path: Path, niche_source: str | None = None) -> int:
    return ingest(
        str(path), is_own_content=True, niche_source=niche_source, deps=make_deps()
    ).video.id


def test_voice_is_measured_from_own_videos(db: Session, sample_video: Path) -> None:
    niche = make_niche(db)
    video_id = own_video(db, sample_video)
    fake = FakeNicheClaude()

    voice = refresh_voice(db, niche, fake, force=True)

    assert voice.video_ids == [video_id]
    assert voice.avg_sentence_words == 10
    assert voice.words_per_second == pytest.approx(10 / 9, abs=0.05)
    assert voice.silences_per_minute > 0
    assert voice.prompt_version == "voice_extract/v1"
    assert load_profile(niche).voice_profile == voice
    assert "Mesuré sur 1 de mes vidéos" in niche.avatar_markdown


def test_voice_keeps_only_what_was_really_said(db: Session, sample_video: Path) -> None:
    niche = make_niche(db)
    own_video(db, sample_video)

    voice = refresh_voice(db, niche, FakeNicheClaude(), force=True)

    assert voice.signature_expressions == ["double your views"]
    assert voice.verbal_tics == []


def test_ingesting_an_own_video_updates_the_voice_automatically(
    db: Session, sample_video: Path
) -> None:
    niche = make_niche(db)
    voice_claude = FakeNicheClaude()
    deps = IngestDeps(
        transcriber=FakeTranscriber(),
        claude=lambda: CombinedClaude(voice_claude),
        downloader=make_deps().downloader,
    )

    ingest(str(sample_video), is_own_content=True, deps=deps)

    assert voice_claude.count("voice_extract") == 1
    db.refresh(niche)
    assert load_profile(niche).voice_profile is not None


class CombinedClaude:
    def __init__(self, voice_claude: FakeNicheClaude) -> None:
        self.voice_claude = voice_claude
        self.ingest_claude = FakeClaude()

    def call_json(self, *, prompt, **kwargs):
        target = self.voice_claude if prompt == "voice_extract" else self.ingest_claude
        return target.call_json(prompt=prompt, **kwargs)


def test_other_peoples_videos_never_touch_the_voice(db: Session, sample_video: Path) -> None:
    niche = make_niche(db)
    ingest(str(sample_video), deps=make_deps())

    assert refresh_voice(db, niche, FakeNicheClaude()) is None
    db.refresh(niche)
    assert load_profile(niche).voice_profile is None


def test_own_video_of_another_niche_is_ignored(db: Session, sample_video: Path) -> None:
    niche = make_niche(db)
    own_video(db, sample_video, niche_source="Autre niche")

    assert refresh_voice(db, niche, FakeNicheClaude()) is None


def test_own_video_without_speech_is_ignored(db: Session, silent_video: Path) -> None:
    niche = make_niche(db)
    own_video(db, silent_video)

    assert refresh_voice(db, niche, FakeNicheClaude()) is None


def test_voice_is_not_repaid_when_nothing_changed(db: Session, sample_video: Path) -> None:
    niche = make_niche(db)
    own_video(db, sample_video)
    fake = FakeNicheClaude()
    refresh_voice(db, niche, fake, force=True)

    refresh_voice(db, niche, fake)

    assert fake.count("voice_extract") == 1


def test_voice_failure_does_not_fail_the_ingestion(db: Session, sample_video: Path) -> None:
    make_niche(db)

    class Failing(CombinedClaude):
        def call_json(self, *, prompt, **kwargs):
            if prompt == "voice_extract":
                raise RuntimeError("Panne simulée")
            return super().call_json(prompt=prompt, **kwargs)

    deps = IngestDeps(
        transcriber=FakeTranscriber(),
        claude=lambda: Failing(FakeNicheClaude()),
        downloader=make_deps().downloader,
    )

    bundle = ingest(str(sample_video), is_own_content=True, deps=deps)

    assert bundle.video.ingested_at is not None


def test_new_own_video_refreshes_the_voice(db: Session, sample_video: Path, tmp_path: Path) -> None:
    niche = make_niche(db)
    first = own_video(db, sample_video)
    second = own_video(db, make_video(tmp_path / "b.mp4", "white"))

    db.refresh(niche)
    voice = refresh_voice(db, niche, FakeNicheClaude())

    assert voice.video_ids == [first, second]


client = TestClient(app)


@pytest.fixture
def api_claude():
    fake = FakeNicheClaude(["Q1", "Q2", None])
    app.dependency_overrides[get_claude] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


def test_interview_through_the_api(db: Session, api_claude: FakeNicheClaude) -> None:
    created = client.post("/niches/interviews", json={})
    assert created.status_code == 201
    interview_id = created.json()["id"]
    assert created.json()["question"] == "Q1"

    second = client.post(f"/niches/interviews/{interview_id}/answers", json={"answer": "R1"})
    assert second.json()["question"] == "Q2"
    assert client.get(f"/niches/interviews/{interview_id}").json()["question"] == "Q2"

    done = client.post(f"/niches/interviews/{interview_id}/answers", json={"answer": "R2"}).json()
    assert done["status"] == "done"
    assert done["question"] is None

    niche = client.get(f"/niches/{done['niche_id']}").json()
    assert niche["profile"]["identity"]["goal"] == "leads"
    assert niche["avatar_markdown"].startswith("# Fiche avatar")
    assert niche["topic_count"] == 50
    assert [item["name"] for item in client.get("/niches").json()] == [niche["name"]]

    again = client.post(f"/niches/interviews/{interview_id}/answers", json={"answer": "R3"})
    assert again.status_code == 409


def test_finish_route(db: Session, api_claude: FakeNicheClaude) -> None:
    interview_id = client.post("/niches/interviews", json={}).json()["id"]

    assert client.post(f"/niches/interviews/{interview_id}/finish").status_code == 409

    client.post(f"/niches/interviews/{interview_id}/answers", json={"answer": "R1"})
    body = client.post(f"/niches/interviews/{interview_id}/finish").json()

    assert body["status"] == "done"


def test_patch_niche_edits_and_versions(db: Session) -> None:
    niche = make_niche(db)

    response = client.patch(
        f"/niches/{niche.id}",
        json={"profile": {"voice": {"tone": "Baveux"}}, "change_note": "Plus de mordant"},
    )

    assert response.status_code == 200
    assert response.json()["profile"]["voice"]["tone"] == "Baveux"
    history = client.get(f"/niches/{niche.id}/versions").json()
    assert history[0]["change_note"] == "Plus de mordant"
    assert history[0]["profile"]["voice"]["tone"] == "Baveux"


def test_patch_errors(db: Session) -> None:
    niche = make_niche(db)
    other = make_niche(db)

    bad = client.patch(f"/niches/{niche.id}", json={"profile": {"identity": {"goal": "gloire"}}})
    duplicate = client.patch(f"/niches/{other.id}", json={"name": niche.name})

    assert bad.status_code == 422
    assert duplicate.status_code == 409
    assert client.patch("/niches/999", json={}).status_code == 404


def test_topic_routes(db: Session, api_claude: FakeNicheClaude) -> None:
    niche = make_niche(db)
    topic = niche.topics[0]["id"]

    marked = client.patch(f"/niches/{niche.id}/topics/{topic}", json={"status": "used"})
    assert marked.json()["status"] == "used"
    assert client.get(f"/niches/{niche.id}").json()["topic_count"] == 49

    api_claude.topic_offset = 500
    refreshed = client.post(f"/niches/{niche.id}/topics").json()
    assert refreshed["topic_count"] == 50

    bad = client.patch(f"/niches/{niche.id}/topics/{topic}", json={"status": "faux"})
    assert bad.status_code == 422
    missing = client.patch(f"/niches/{niche.id}/topics/nope", json={"status": "used"})
    assert missing.status_code == 404


def test_voice_route(db: Session, sample_video: Path, api_claude: FakeNicheClaude) -> None:
    niche = make_niche(db)

    assert client.post(f"/niches/{niche.id}/voice").json() is None

    own_video(db, sample_video)
    body = client.post(f"/niches/{niche.id}/voice?force=true").json()

    assert body["avg_sentence_words"] == 10


runner = CliRunner()


@pytest.fixture
def cli_claude(monkeypatch: pytest.MonkeyPatch) -> FakeNicheClaude:
    fake = FakeNicheClaude(["Q1", "Q2", None])
    monkeypatch.setattr(cli_module, "default_claude", lambda: fake)
    return fake


def test_cli_interview(db: Session, cli_claude: FakeNicheClaude) -> None:
    result = runner.invoke(cli, ["niche", "new"], input="R1\nR2\n")

    assert result.exit_code == 0, result.output
    assert "Q1" in result.output and "Q2" in result.output
    assert "50 sujet(s)" in result.output
    assert count(db, Niche) == 1


def test_cli_early_finish(db: Session, cli_claude: FakeNicheClaude) -> None:
    result = runner.invoke(cli, ["niche", "new"], input="R1\n/fin\n")

    assert result.exit_code == 0, result.output
    assert cli_claude.count("niche_profile") == 1


def test_cli_resume(db: Session, cli_claude: FakeNicheClaude) -> None:
    interview = start_interview(db, cli_claude)

    result = runner.invoke(cli, ["niche", "new", "--resume", str(interview.id)], input="R1\nR2\n")

    assert result.exit_code == 0, result.output
    assert count(db, NicheInterview) == 1


def test_cli_show_list_topics(db: Session, cli_claude: FakeNicheClaude) -> None:
    niche = make_niche(db)

    assert niche.name in runner.invoke(cli, ["niche", "list"]).output
    assert "Fiche avatar" in runner.invoke(cli, ["niche", "show", str(niche.id)]).output
    topics = runner.invoke(cli, ["niche", "topics", str(niche.id), "--limit", "3"]).output
    assert len(topics.strip().splitlines()) == 3
    assert topics.startswith("10/10")
    all_topics = runner.invoke(cli, ["niche", "topics", str(niche.id)]).output
    assert all_topics.count("(hypothèse)") == 25
    assert runner.invoke(cli, ["niche", "show", "999"]).exit_code == 1


def test_cli_voice(db: Session, sample_video: Path, cli_claude: FakeNicheClaude) -> None:
    niche = make_niche(db)

    empty = runner.invoke(cli, ["niche", "voice", str(niche.id)])
    assert "Aucune de mes vidéos" in empty.output

    own_video(db, sample_video)
    result = runner.invoke(cli, ["niche", "voice", str(niche.id), "--force"])
    assert "1 vidéo(s)" in result.output


@pytest.mark.parametrize(
    ("name", "variables"),
    [
        ("niche_interview", {"transcript": "[]"}),
        ("niche_profile", {"transcript": "[]"}),
        ("niche_topics", {"count": 50, "profile": "{}", "existing": "[]"}),
        ("voice_extract", {"transcripts": "[]", "stats": "{}", "phrases": "[]"}),
    ],
)
def test_prompts_render(name: str, variables: dict) -> None:
    assert "{{" not in load_prompt(name, "v1").render_user(variables)


def test_niche_without_profile_stays_valid(db: Session) -> None:
    niche = Niche(name="Ancienne")
    db.add(niche)
    db.commit()

    assert niche.topics == []
    assert niche.version == 0
    assert client.get(f"/niches/{niche.id}").json()["profile"] is None
    assert client.post(f"/niches/{niche.id}/voice").status_code == 409


def test_versions_cascade_with_the_niche(db: Session) -> None:
    niche = make_niche(db)
    assert count(db, NicheVersion) >= 2

    db.execute(delete(Niche).where(Niche.id == niche.id))
    db.commit()

    assert count(db, NicheVersion) == 0
