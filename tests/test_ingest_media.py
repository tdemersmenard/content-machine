"""ffmpeg et yt-dlp pour de vrai, sur une vidéo générée localement."""

import functools
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app.ingest.acquire import AcquireError, download_url
from app.ingest.media import (
    MediaError,
    detect_scene_cuts,
    extract_audio,
    extract_frame,
    probe,
    sha256_file,
)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        pass


@pytest.fixture
def local_server(sample_video: Path) -> Iterator[str]:
    handler = functools.partial(QuietHandler, directory=str(sample_video.parent))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_probe(sample_video: Path, silent_video: Path) -> None:
    info = probe(sample_video)

    assert info.duration_s == pytest.approx(9, abs=0.2)
    assert info.has_video and info.has_audio
    assert probe(silent_video).has_audio is False


def test_probe_rejects_a_file_that_is_not_a_video(tmp_path: Path) -> None:
    fake = tmp_path / "fake.mp4"
    fake.write_text("pas une vidéo")

    with pytest.raises(MediaError):
        probe(fake)


def test_scene_detection_finds_the_real_cuts(sample_video: Path) -> None:
    assert detect_scene_cuts(sample_video, 0.3) == pytest.approx([3.0, 6.0], abs=0.1)


def test_high_threshold_finds_nothing_on_a_static_video_part(sample_video: Path) -> None:
    assert detect_scene_cuts(sample_video, 1.0) == []


def test_extract_frame_writes_an_image(sample_video: Path, tmp_path: Path) -> None:
    output = tmp_path / "frame.jpg"

    extract_frame(sample_video, 4.0, output)

    assert output.read_bytes()[:2] == b"\xff\xd8"  # Signature JPEG.


def test_extract_audio_produces_a_16khz_track(sample_video: Path, tmp_path: Path) -> None:
    output = tmp_path / "audio.wav"

    extract_audio(sample_video, output)

    assert probe(output).has_audio


def test_hash_is_stable_and_depends_on_content(sample_video: Path, silent_video: Path) -> None:
    assert sha256_file(sample_video) == sha256_file(sample_video)
    assert sha256_file(sample_video) != sha256_file(silent_video)


def test_yt_dlp_downloads_from_a_local_server_without_inventing_stats(
    local_server: str, sample_video: Path, tmp_path: Path
) -> None:
    downloaded = download_url(f"{local_server}/{sample_video.name}", tmp_path)

    assert sha256_file(downloaded.path) == sha256_file(sample_video)
    for stat in ("views", "likes", "comments", "shares", "saves"):
        assert downloaded.metadata[stat] is None


def test_yt_dlp_failure_raises_acquire_error(local_server: str, tmp_path: Path) -> None:
    with pytest.raises(AcquireError):
        download_url(f"{local_server}/introuvable.mp4", tmp_path)
