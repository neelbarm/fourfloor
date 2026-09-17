"""``POST /api/fetch``: a pasted link becomes a source, over the real server.

The fake yt-dlp from :mod:`tests.test_fetch` goes on PATH for the whole module,
so this drives the same code the browser does -- start the job, follow the SSE
stream, and check the result is byte-for-byte the shape an upload returns, which
is what lets the page join the normal path at the controls screen.
"""

from __future__ import annotations

import os
import threading

import pytest

from fourfloor import store
from fourfloor.server import make_server
from tests.test_fetch import FAKE, FIXTURE, url_for      # noqa: F401 - shared fake
from tests.test_server import Client


@pytest.fixture(scope="module")
def fake_path(tmp_path_factory):
    """The fake yt-dlp on PATH for this module, and put back afterwards."""
    import stat
    import sys

    if not FIXTURE.is_file():
        pytest.skip("fixture missing")
    d = tmp_path_factory.mktemp("fake-bin-http")
    body = d / "fake_ytdlp.py"
    body.write_text(FAKE, encoding="utf8")
    script = d / "yt-dlp"
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{body}" "$@"\n',
                      encoding="utf8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    old_path, old_audio = os.environ.get("PATH", ""), os.environ.get("FAKE_YTDLP_AUDIO")
    old_override = os.environ.pop("FOURFLOOR_YTDLP", None)
    os.environ["PATH"] = f"{d}{os.pathsep}{old_path}"
    os.environ["FAKE_YTDLP_AUDIO"] = str(FIXTURE)
    yield d
    os.environ["PATH"] = old_path
    if old_audio is None:
        os.environ.pop("FAKE_YTDLP_AUDIO", None)
    else:
        os.environ["FAKE_YTDLP_AUDIO"] = old_audio
    if old_override is not None:
        os.environ["FOURFLOOR_YTDLP"] = old_override


@pytest.fixture(scope="module")
def app(fake_path, tmp_path_factory):
    srv = make_server(0, tmp_path_factory.mktemp("fourfloor-fetch-home"))
    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05},
                              daemon=True)
    thread.start()
    client = Client(f"http://127.0.0.1:{srv.server_address[1]}")
    client.lib = srv.app.lib                         # for the on-disk assertions
    yield client
    srv.shutdown()
    srv.server_close()
    srv.app.close()


def test_the_config_says_whether_links_can_be_fetched(app) -> None:
    status, cfg = app.json("/api/config")
    assert status == 200
    assert cfg["fetch"] is True                      # the fake is on PATH


@pytest.mark.parametrize("url, needle", [
    ("file:///etc/passwd", "http"),
    ("http://127.0.0.1:4444/api/config", "local address"),
    ("http://169.254.169.254/latest/meta-data/", "local address"),
    ("http://nas.local/song.mp3", "local address"),
    ("", "paste a link"),
])
def test_a_link_the_server_will_not_follow_is_a_400(app, url, needle) -> None:
    status, out = app.json("/api/fetch", "POST", {"url": url})
    assert status == 400, out
    assert needle in out["error"], out


def test_a_bad_body_is_refused_before_anything_is_started(app) -> None:
    status, out = app.json("/api/fetch", "POST", {})
    assert status == 400 and "paste a link" in out["error"]


def test_a_pasted_link_becomes_a_source_the_way_an_upload_does(app) -> None:
    """The whole path: POST, follow the stream, land on an analysed source."""
    status, started = app.json("/api/fetch", "POST",
                               {"url": url_for("Pasted Link")})
    assert status == 200, started
    assert store.safe_id(started["job"])
    assert started["site"] == "example.com"

    events = app.events(started["job"])
    kinds = [e["type"] for e in events]
    assert kinds[0] == "queued" and kinds[-1] == "done", kinds
    assert "error" not in kinds, [e for e in events if e["type"] == "error"]

    # the percentages yt-dlp printed reached the browser
    pcts = [e["percent"] for e in events if e["type"] == "progress"]
    assert pcts and pcts[-1] == 100.0 and pcts == sorted(pcts)
    assert [e["name"] for e in events if e["type"] == "phase"] == ["fetch", "fetch",
                                                                  "analyse"]

    meta = events[-1]["result"]
    assert store.safe_id(meta["id"])
    assert meta["name"] == "Pasted Link.mp3"
    assert meta["title"] == "Pasted Link"
    assert 60 < meta["analysis"]["tempo"]["bpm"] < 200
    assert meta["analysis"]["key"]["camelot"] and meta["analysis"]["sections"]
    assert len(meta["wave"]) == 900                  # exactly what an upload returns
    assert meta["link"]["url"].startswith("https://example.com/")
    assert meta["link"]["title"] == "Pasted Link"
    assert meta["link"]["duration"] == 137.0

    # …and it is a source a remix can be built from, with no upload involved
    status, remix = app.json("/api/remix", "POST", {
        "source": meta["id"], "bpm": "124", "length": "1:00", "form": "radio"})
    assert status == 200, remix
    events = app.events(remix["job"])
    assert events[-1]["type"] == "done", events[-1]
    rid = events[-1]["result"]["id"]
    status, headers, mp3 = app.request(f"/api/remixes/{rid}/remix.mp3")
    assert status == 200 and len(mp3) > 50_000


def test_nothing_of_the_download_is_left_behind(app) -> None:
    status, started = app.json("/api/fetch", "POST", {"url": url_for("Tidy")})
    assert status == 200
    events = app.events(started["job"])
    assert events[-1]["type"] == "done", events[-1]
    assert not list(app.lib.uploads.iterdir()), "a download temporary survived"
    sid = events[-1]["result"]["id"]
    assert app.lib.source_audio(sid).name == "source.mp3"


def test_a_dead_link_fails_the_job_with_a_sentence(app) -> None:
    status, started = app.json("/api/fetch", "POST",
                               {"url": "https://example.com/nope"})
    assert status == 200, started
    events = app.events(started["job"])
    assert events[-1]["type"] == "error", events[-1]
    assert "not available" in events[-1]["message"]


def test_a_playlist_is_refused_with_a_way_forward(app) -> None:
    status, started = app.json(
        "/api/fetch", "POST",
        {"url": "https://example.com/watch?title=Set&count=4&playlist=1"})
    assert status == 200, started
    events = app.events(started["job"])
    assert events[-1]["type"] == "error"
    message = events[-1]["message"]
    assert "4 tracks" in message and "fourfloor fetch" in message


def test_the_page_offers_the_link_field(app) -> None:
    body = app.request("/")[2].decode("utf8")
    assert 'id="linkInput"' in body
    assert "SoundCloud" in body
    js = app.request("/app.js")[2].decode("utf8")
    assert "/api/fetch" in js
