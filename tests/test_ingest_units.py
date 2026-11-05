"""Fonctions pures de l'ingestion : plans, signaux, métadonnées, timeline."""

from datetime import UTC, datetime

import pytest

from app.ingest.acquire import extract_hashtags, metadata_from_info
from app.ingest.media import parse_scene_times
from app.ingest.schemas import Word
from app.ingest.shots import build_shots, clean_cuts, forced_timestamps, plan_frames
from app.ingest.signals import compute_signals, find_silences, word_rate_buckets
from app.ingest.timeline import group_speech


def word(text: str, start: float, end: float) -> Word:
    return Word(word=text, start=start, end=end)


def test_forced_frames_every_second_in_hook_then_every_three() -> None:
    assert forced_timestamps(15) == [0, 1, 2, 3, 4, 5, 8, 11, 14]


def test_forced_frames_never_go_past_the_end() -> None:
    assert forced_timestamps(2.5) == [0, 1, 2]
    assert forced_timestamps(3.0) == [0, 1, 2]


def test_clean_cuts_sorts_and_drops_edges_and_duplicates() -> None:
    assert clean_cuts([6.0, 0.0, 3.0, 3.04, 9.0], 9.0) == [3.0, 6.0]


def test_build_shots_covers_the_whole_video() -> None:
    assert build_shots([3.0, 6.0], 9.0) == [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]
    assert build_shots([], 9.0) == [(0.0, 9.0)]


def test_plan_merges_forced_frames_close_to_a_cut() -> None:
    planned = plan_frames([3.1], 7.0)

    assert [frame.t_s for frame in planned] == [0, 1, 2, 3.1, 4, 5]
    assert [frame.reason for frame in planned] == [
        "hook",
        "hook",
        "hook",
        "scene",
        "hook",
        "hook",
    ]
    assert [frame.shot_index for frame in planned] == [0, 0, 0, 1, 1, 1]


def test_plan_labels_frames_after_the_hook_as_interval() -> None:
    assert plan_frames([], 9.0)[-1].reason == "interval"


def test_parse_scene_times() -> None:
    output = "[Parsed_showinfo_1] n:0 pts:75 pts_time:3 pos:1\nn:1 pts_time:6.04 foo\n"

    assert parse_scene_times(output) == [3.0, 6.04]


def test_silences_include_start_middle_and_end() -> None:
    words = [word("a", 1.0, 1.2), word("b", 1.3, 1.5), word("c", 2.0, 2.2)]

    silences = find_silences(words, 3.0)

    assert [(s.start_s, s.end_s) for s in silences] == [(0.0, 1.0), (1.5, 2.0), (2.2, 3.0)]


def test_gap_of_exactly_the_threshold_is_not_a_silence() -> None:
    assert find_silences([word("a", 0.0, 1.0), word("b", 1.4, 2.0)], 2.0) == []


def test_no_speech_is_one_long_silence() -> None:
    assert [s.duration_s for s in find_silences([], 5.0)] == [5.0]


def test_word_rate_buckets_use_the_real_length_of_the_last_bucket() -> None:
    words = [word("a", 0.5, 0.6), word("b", 2.9, 3.1), word("c", 4.0, 4.1)]

    buckets = word_rate_buckets(words, 4.5)

    assert [(b.start_s, b.end_s, b.words) for b in buckets] == [(0, 3, 2), (3, 4.5, 1)]
    assert buckets[1].words_per_second == pytest.approx(1 / 1.5, abs=0.001)


def test_compute_signals() -> None:
    words = [word("a", 0.0, 0.5), word("b", 0.5, 1.0)]
    shots = [(0.0, 1.0), (1.0, 2.5), (2.5, 4.0), (4.0, 10.0)]

    signals = compute_signals(words, shots, 10.0, None)

    assert signals.words_per_second == 0.2
    assert signals.shot_count == 4
    assert signals.avg_shot_duration_s == 2.5
    assert signals.cuts_first_3s == 2
    assert signals.promise is None


def test_group_speech_splits_on_silences() -> None:
    words = [word("a", 0.0, 0.5), word("b", 0.6, 1.0), word("c", 2.0, 2.5)]

    assert [[w.word for w in group] for group in group_speech(words)] == [["a", "b"], ["c"]]


def test_hashtags_are_unique_and_support_accents() -> None:
    assert extract_hashtags("Top #été #Fitness et #été") == ["été", "Fitness"]
    assert extract_hashtags(None) == []


def test_metadata_keeps_missing_stats_as_none() -> None:
    metadata = metadata_from_info({"extractor_key": "TikTok", "view_count": 0})

    assert metadata["platform"] == "tiktok"
    assert metadata["views"] == 0  # Un vrai zéro reste un zéro.
    for missing in ("likes", "comments", "shares", "saves", "creator", "posted_at", "sound"):
        assert metadata[missing] is None


def test_metadata_maps_every_available_field() -> None:
    metadata = metadata_from_info(
        {
            "extractor_key": "Youtube",
            "uploader": "Ada",
            "timestamp": 1_750_000_000,
            "view_count": 10,
            "like_count": 5,
            "comment_count": 2,
            "repost_count": 1,
            "save_count": 3,
            "description": "Salut #viral",
            "tags": ["viral", "shorts"],
            "track": "Chanson",
            "artists": ["Artiste"],
        }
    )

    assert metadata["creator"] == "Ada"
    assert metadata["posted_at"] == datetime.fromtimestamp(1_750_000_000, UTC)
    assert (metadata["likes"], metadata["comments"]) == (5, 2)
    assert (metadata["shares"], metadata["saves"]) == (1, 3)
    assert metadata["hashtags"] == ["viral", "shorts"]
    assert metadata["sound"] == "Artiste - Chanson"


def test_upload_date_is_used_when_there_is_no_timestamp() -> None:
    metadata = metadata_from_info({"upload_date": "20260315"})

    assert metadata["posted_at"] == datetime(2026, 3, 15, tzinfo=UTC)
