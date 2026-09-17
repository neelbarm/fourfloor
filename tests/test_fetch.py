"""Link fetching: URL rules, naming, playlists, pairs, and yt-dlp's failures.

Nothing here touches the network. A fake ``yt-dlp`` goes on PATH -- a small
Python script that answers ``--dump-single-json`` with metadata derived from
the URL, prints the same ``[download]  12.3%`` lines the real one prints, and
copies the bundled fixture into the ``-o`` template. That covers the whole
contract fourfloor depends on: the JSON probe, the progress format, where the
file lands and what a failure looks like.

The one test that does use the network is at the bottom, skipped unless
``FOURFLOOR_NET_TESTS=1``.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from fourfloor import fetch

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lofi-7.mp3"

# ---------------------------------------------------------------------------
# the fake binary
# ---------------------------------------------------------------------------

FAKE = '''\
"""A stand-in for yt-dlp: enough of its surface for fourfloor to be tested."""
import json, os, re, shutil, sys, time
from urllib.parse import urlsplit, parse_qs

AUDIO = os.environ["FAKE_YTDLP_AUDIO"]
LOG = os.environ.get("FAKE_YTDLP_LOG")

argv = sys.argv[1:]
url = argv[-1]
q = parse_qs(urlsplit(url).query)
path = urlsplit(url).path
title = q.get("title", [path.rstrip("/").split("/")[-1] or "untitled"])[0]
count = int(q.get("count", ["3"])[0])

if LOG:
    with open(LOG, "a", encoding="utf8") as fh:
        fh.write(" ".join(argv) + "\\n")

if "/nope" in path:
    sys.stderr.write("ERROR: [youtube] nope: Video unavailable\\n")
    sys.exit(1)
if "/boom" in path:
    sys.stderr.write("ERROR: Unsupported URL: " + url + "\\n")
    sys.exit(1)
if "/hang" in path:
    time.sleep(30)
    sys.exit(0)

is_set = "/sets/" in path or "playlist" in q

if "--dump-single-json" in argv:
    if is_set:
        print(json.dumps({
            "_type": "playlist", "title": title, "id": "set1",
            "entries": [{"_type": "url", "id": "e%d" % i, "title": "%s %d" % (title, i),
                         "duration": 100 + i,
                         "url": "https://example.com/watch?title=%s+%d" % (title, i)}
                        for i in range(1, count + 1)],
        }))
    else:
        print(json.dumps({
            "_type": "video", "id": "abc123", "title": title,
            "uploader": "Fake Uploader", "duration": 137.0,
            "webpage_url": url, "extractor_key": "Fake",
        }))
    sys.exit(0)

# a download: honour -o, print progress, leave an mp3 behind
out = argv[argv.index("-o") + 1] if "-o" in argv else "audio.%(ext)s"
for pct in ("0.0", "12.5", "48.2", "100.0"):
    print("[download] %5s%% of  3.44MiB at 1.00MiB/s ETA 00:0%d" % (pct, 1))
    sys.stdout.flush()
target = out.replace("%(ext)s", "mp3")
os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
shutil.copyfile(AUDIO, target)
print("[ExtractAudio] Destination: " + target)
'''


@pytest.fixture(scope="session")
def fake_bin(tmp_path_factory) -> Path:
    """A directory holding an executable fake ``yt-dlp``."""
    if not FIXTURE.is_file():
        pytest.skip("fixture missing")
    d = tmp_path_factory.mktemp("fake-bin")
    body = d / "fake_ytdlp.py"
    body.write_text(FAKE, encoding="utf8")
    # a shell wrapper rather than a shebang: this checkout's interpreter path
    # has a space in it, and a shebang line cannot quote
    script = d / "yt-dlp"
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{body}" "$@"\n',
                      encoding="utf8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return d


@pytest.fixture
def fake(fake_bin, monkeypatch):
    """Put the fake first on PATH for one test."""
    monkeypatch.delenv("FOURFLOOR_YTDLP", raising=False)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_YTDLP_AUDIO", str(FIXTURE))
    return fake_bin


def url_for(title: str, **extra) -> str:
    q = "&".join([f"title={title.replace(' ', '+')}"]
                 + [f"{k}={v}" for k, v in extra.items()])
    return f"https://example.com/watch?{q}"


# ---------------------------------------------------------------------------
# urls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://music.youtube.com/watch?v=x",
    "https://soundcloud.com/artist/track",
    "https://soundcloud.com/artist/sets/a-set",
    "http://example.com/a.mp3",
])
def test_a_public_link_is_accepted(url) -> None:
    assert fetch.check_url(url) == url


@pytest.mark.parametrize("url, needle", [
    ("file:///etc/passwd", "http"),
    ("ftp://example.com/x.mp3", "http"),
    ("/Users/me/song.mp3", "http"),
    ("", "paste a link"),
    ("   ", "paste a link"),
    ("http://localhost:4444/api/config", "local address"),
    ("http://127.0.0.1/x", "local address"),
    ("http://[::1]/x", "local address"),
    ("http://169.254.169.254/latest/meta-data/", "local address"),
    ("http://192.168.1.10/x", "local address"),
    ("http://10.0.0.5/x", "local address"),
    ("http://172.16.4.4/x", "local address"),
    ("http://nas.local/song.mp3", "local address"),
    ("http://router/x", "local address"),
    ("https://example.com/\nHost: evil", "control characters"),
    ("https://" + "a" * 3000, "implausibly long"),
])
def test_a_link_we_will_not_follow_is_refused(url, needle) -> None:
    with pytest.raises(fetch.FetchError) as exc:
        fetch.check_url(url)
    assert needle in str(exc.value)


def test_the_site_label_reads_like_a_person_wrote_it() -> None:
    assert fetch.site_of("https://youtu.be/x") == "youtube"
    assert fetch.site_of("https://music.youtube.com/watch?v=x") == "youtube music"
    assert fetch.site_of("https://soundcloud.com/a/b") == "soundcloud"
    assert fetch.site_of("https://example.org/a") == "example.org"


# ---------------------------------------------------------------------------
# names on disk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expect", [
    ("Simple Title", "Simple Title"),
    ("AC/DC — Back in Black", "AC DC Back in Black"),
    ("../../etc/passwd", "etc passwd"),
    ("a\x00b", "a b"),
    ("...", "track"),
    ("", "track"),
    ("  spaced   out  ", "spaced out"),
    ("Tame Impala - Let It Happen (Soulwax Remix)",
     "Tame Impala - Let It Happen (Soulwax Remix)"),
])
def test_a_title_becomes_a_filename_we_chose(raw, expect) -> None:
    got = fetch.sanitize(raw)
    assert got == expect
    assert "/" not in got and "\x00" not in got
    assert not got.startswith(".")


def test_a_long_title_is_cut_to_something_typeable() -> None:
    assert len(fetch.sanitize("x" * 400)) == 80


def test_pair_stems_follow_the_learn_convention() -> None:
    assert fetch.pair_stem("Midnight City", "original") == "Midnight City.original"
    assert fetch.pair_stem("Midnight City", "remix") == "Midnight City.remix"
    assert fetch.pair_stem("Midnight City", None) == "Midnight City"
    with pytest.raises(fetch.FetchError):
        fetch.pair_stem("x", "instrumental")


def test_destinations_resolve_to_the_reference_folders(tmp_path) -> None:
    assert fetch.resolve_dest("remixes") == fetch.REMIXES_DIR
    assert fetch.resolve_dest(None) == fetch.REMIXES_DIR
    assert fetch.resolve_dest("pairs") == fetch.PAIRS_DIR
    assert fetch.resolve_dest(str(tmp_path)) == tmp_path


def test_free_path_never_picks_a_name_that_exists(tmp_path) -> None:
    (tmp_path / "song.mp3").write_bytes(b"x")
    assert fetch.free_path(tmp_path, "song").name == "song-2.mp3"
    (tmp_path / "song-2.mp3").write_bytes(b"x")
    assert fetch.free_path(tmp_path, "song").name == "song-3.mp3"


# ---------------------------------------------------------------------------
# fetching, with the fake
# ---------------------------------------------------------------------------

def test_one_track_lands_with_its_metadata(fake, tmp_path) -> None:
    seen = []
    got = fetch.fetch(url_for("Night Drive"), tmp_path,
                      progress=lambda pct, note: seen.append((pct, note)))

    assert got.path == tmp_path / "Night Drive.mp3"
    assert got.path.read_bytes() == FIXTURE.read_bytes()
    assert got.title == "Night Drive"
    assert got.uploader == "Fake Uploader"
    assert got.duration == 137.0
    assert got.site == "example.com"
    assert got.bytes == FIXTURE.stat().st_size

    # the percentages yt-dlp printed came through the callback, in order
    percents = [p for p, _ in seen]
    assert percents[0] == 0.0 and percents[-1] == 100.0
    assert 48.2 in percents
    assert percents == sorted(percents)
    assert os.fspath(got) == str(got.path)          # usable anywhere a path is


def test_the_download_is_an_mp3_the_analyser_can_read(fake, tmp_path) -> None:
    from fourfloor.audio import decode

    got = fetch.fetch(url_for("Readable"), tmp_path)
    assert decode(got).duration > 1.0               # os.PathLike, straight in


def test_a_second_copy_does_not_overwrite_the_first(fake, tmp_path) -> None:
    a = fetch.fetch(url_for("Same Song"), tmp_path)
    b = fetch.fetch(url_for("Same Song"), tmp_path)
    c = fetch.fetch(url_for("Same Song"), tmp_path)
    assert [p.path.name for p in (a, b, c)] == [
        "Same Song.mp3", "Same Song-2.mp3", "Same Song-3.mp3"]
    assert a.path.exists() and b.path.exists()


def test_a_name_and_a_kind_write_half_a_pair(fake, tmp_path) -> None:
    got = fetch.fetch(url_for("Whatever The Site Calls It"), tmp_path,
                      name="midnight city", kind="original")
    assert got.path.name == "midnight city.original.mp3"


def test_a_pair_lands_as_original_and_remix(fake, tmp_path) -> None:
    notes = []
    got = fetch.fetch_pair(url_for("The Original"), url_for("The Remix"),
                           "midnight city", dest_dir=tmp_path,
                           progress=lambda pct, note: notes.append(note))
    assert [f.path.name for f in got] == ["midnight city.original.mp3",
                                          "midnight city.remix.mp3"]
    assert [f.title for f in got] == ["The Original", "The Remix"]
    assert any(n.startswith("original") for n in notes)
    assert any(n.startswith("remix") for n in notes)


def test_a_playlist_brings_back_every_track(fake, tmp_path) -> None:
    notes = []
    got = fetch.fetch_all(url_for("Deep House", count=3) + "&playlist=1", tmp_path,
                          progress=lambda pct, note: notes.append(note))
    assert len(got) == 3
    assert [f.path.name for f in got] == ["Deep House 1.mp3", "Deep House 2.mp3",
                                          "Deep House 3.mp3"]
    assert all(f.playlist == "Deep House" for f in got)
    assert any("1/3" in n for n in notes) and any("3/3" in n for n in notes)


def test_a_named_playlist_numbers_its_files(fake, tmp_path) -> None:
    got = fetch.fetch_all(url_for("Set", count=2) + "&playlist=1", tmp_path, name="refs")
    assert [f.path.name for f in got] == ["refs-01.mp3", "refs-02.mp3"]


def test_a_soundcloud_set_is_a_playlist(fake, tmp_path) -> None:
    info = fetch.probe("https://soundcloud.com/artist/sets/summer?title=Summer&count=2")
    assert fetch.is_playlist(info)
    assert len(fetch.entries_of(info)) == 2
    assert not fetch.is_playlist(fetch.probe(url_for("Single")))


def test_fetching_a_playlist_with_fetch_says_to_use_fetch_all(fake, tmp_path) -> None:
    with pytest.raises(fetch.FetchError) as exc:
        fetch.fetch(url_for("Set", count=4) + "&playlist=1", tmp_path)
    assert "playlist" in str(exc.value) and "4 tracks" in str(exc.value)


def test_a_dead_link_is_reported_in_one_sentence(fake, tmp_path) -> None:
    with pytest.raises(fetch.FetchError) as exc:
        fetch.fetch("https://example.com/nope", tmp_path)
    assert "not available" in str(exc.value)
    assert not list(tmp_path.iterdir()), "a failed fetch left something behind"


def test_a_site_yt_dlp_does_not_know_is_reported(fake, tmp_path) -> None:
    with pytest.raises(fetch.FetchError) as exc:
        fetch.fetch("https://example.com/boom", tmp_path)
    assert "does not know that site" in str(exc.value)


def test_a_download_that_never_ends_is_given_up_on(fake, tmp_path) -> None:
    with pytest.raises(fetch.FetchError) as exc:
        fetch.fetch("https://example.com/hang", tmp_path, timeout=1.5)
    assert "gave up" in str(exc.value)
    assert not list(tmp_path.iterdir())


def test_a_temp_fetch_cleans_up_after_itself(fake) -> None:
    got, cleanup = fetch.fetch_to_temp(url_for("Scratch"))
    folder = got.path.parent
    assert got.path.is_file()
    cleanup()
    assert not folder.exists()


def test_the_command_line_yt_dlp_gets_is_the_one_we_promised(fake, tmp_path) -> None:
    log = tmp_path / "argv.log"
    os.environ["FAKE_YTDLP_LOG"] = str(log)
    try:
        fetch.fetch(url_for("Flags"), tmp_path / "out")
    finally:
        os.environ.pop("FAKE_YTDLP_LOG")
    probe_line, download_line = log.read_text().strip().splitlines()
    assert "--dump-single-json" in probe_line
    for flag in ("--extract-audio", "--audio-format mp3", "--audio-quality 0",
                 "--no-playlist", "--newline"):
        assert flag in download_line, download_line


# ---------------------------------------------------------------------------
# without yt-dlp
# ---------------------------------------------------------------------------

def test_a_missing_yt_dlp_says_how_to_get_it(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("FOURFLOOR_YTDLP", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert not fetch.available()
    with pytest.raises(fetch.FetchError) as exc:
        fetch.fetch("https://example.com/x", tmp_path)
    message = str(exc.value)
    assert "yt-dlp" in message and "install" in message.lower()


def test_the_binary_can_be_pointed_at_explicitly(fake_bin, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("FAKE_YTDLP_AUDIO", str(FIXTURE))
    monkeypatch.setenv("FOURFLOOR_YTDLP", str(fake_bin / "yt-dlp"))
    assert fetch.available()
    assert fetch.fetch(url_for("Explicit"), tmp_path).path.name == "Explicit.mp3"


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def run_cli(argv: list[str], env: dict | None = None):
    """Run the CLI in a subprocess, the way a person runs it."""
    e = {**os.environ, "NO_COLOR": "1", **(env or {})}
    return subprocess.run([sys.executable, "-m", "fourfloor", *argv],
                          capture_output=True, text=True, timeout=300, env=e,
                          cwd=str(Path(__file__).resolve().parents[1]))


def test_fetch_writes_where_it_was_told_and_reports_it(fake, tmp_path) -> None:
    dest = tmp_path / "refs"
    out = run_cli(["fetch", url_for("Cli Track"), "--to", str(dest)])
    assert out.returncode == 0, out.stderr
    assert (dest / "Cli Track.mp3").is_file()
    assert "Cli Track" in out.stdout
    assert "fetched" in out.stdout and str(dest) in out.stdout
    assert "100.0%" in out.stdout                    # the bar reported percent


def test_fetch_json_lists_what_landed(fake, tmp_path) -> None:
    out = run_cli(["fetch", url_for("Json Track"), "--to", str(tmp_path), "--json"])
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert data["dest"] == str(tmp_path)
    row = data["files"][0]
    assert row["title"] == "Json Track" and row["duration"] == 137.0
    assert Path(row["path"]).is_file()


def test_fetch_writes_a_pair_with_the_right_names(fake, tmp_path) -> None:
    out = run_cli(["fetch", "--original", url_for("Orig"), "--remix", url_for("Rmx"),
                   "--name", "midnight city", "--to", str(tmp_path), "--json"])
    assert out.returncode == 0, out.stderr
    names = sorted(Path(f["path"]).name for f in json.loads(out.stdout)["files"])
    assert names == ["midnight city.original.mp3", "midnight city.remix.mp3"]


@pytest.mark.parametrize("argv, needle", [
    (["fetch"], "give me a link"),
    (["fetch", "--original", "https://example.com/a", "--name", "x"], "a pair needs"),
    (["fetch", "https://example.com/a", "--as", "remix"], "--name"),
    (["fetch", "file:///etc/passwd"], "http"),
    (["fetch", "http://127.0.0.1/x"], "local address"),
    (["inspect"], "give me a file"),
    (["remix"], "give me a file"),
])
def test_a_command_line_that_makes_no_sense_is_refused(fake, argv, needle) -> None:
    out = run_cli(argv)
    assert out.returncode != 0
    assert needle in out.stderr, out.stderr


def test_inspect_reads_a_link_and_leaves_nothing_behind(fake, tmp_path) -> None:
    # its own TMPDIR, so "nothing behind" is about this run and nothing else
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    out = run_cli(["inspect", "--url", url_for("Inspected")], {"TMPDIR": str(scratch)})
    assert out.returncode == 0, out.stderr
    assert "tempo" in out.stdout and "BPM" in out.stdout
    assert "Inspected" in out.stdout
    assert not list(scratch.iterdir()), "the fetched file outlived the command"


def test_remix_from_a_link_writes_the_remix_it_was_asked_for(fake, tmp_path) -> None:
    out_path = tmp_path / "linked.house.mp3"
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    out = run_cli(["remix", "--url", url_for("Linked"), "-o", str(out_path),
                   "--length", "1:00", "--bpm", "124", "--no-wav"],
                  {"TMPDIR": str(scratch)})
    assert out.returncode == 0, out.stderr
    assert out_path.is_file() and out_path.stat().st_size > 10_000
    assert not list(scratch.iterdir()), "the fetched file outlived the command"


# ---------------------------------------------------------------------------
# the real thing
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("FOURFLOOR_NET_TESTS") != "1",
                    reason="set FOURFLOOR_NET_TESTS=1 to fetch over the network")
def test_a_real_creative_commons_clip_comes_back(tmp_path) -> None:
    """The one test that talks to the internet. Nothing it downloads is kept."""
    if not shutil.which("yt-dlp"):
        pytest.skip("yt-dlp is not installed")
    # Blender's "Caminandes 3: Llamigos": 2:30, CC-BY, and it has been up for
    # a decade. Override with FOURFLOOR_NET_URL if it ever is not.
    url = os.environ.get("FOURFLOOR_NET_URL",
                         "https://www.youtube.com/watch?v=SkVqJ1SGeL0")
    got = fetch.fetch(url, tmp_path, timeout=300)
    assert got.path.is_file() and got.bytes > 20_000
    assert got.title and got.duration > 0
    print(f"\nfetched {got.title!r} by {got.uploader!r}, "
          f"{got.duration:.0f}s, {fetch.fmt_bytes(got.bytes)} -> {got.path.name}")

    from fourfloor.audio import decode
    assert decode(got).duration == pytest.approx(got.duration, abs=2.0)
