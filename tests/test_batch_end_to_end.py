"""One real batch, driven through the command line, on two short tracks.

This is the only test here that runs the actual engine, so it stays small: two
copies of the bundled fixture at ``--length 1:00``, which the club form rounds
up to its 56-bar minimum. The names carry the spaces, the em dash and the
accent that break file URLs, because that is what a real music folder looks
like and what Rekordbox refuses to import when it is got wrong.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import pytest

from fourfloor import cli, export

TARGET_BPM = 124.0
NAMES = ("01 Nuit Blanche — Édition.mp3", "02 second track.mp3")

pytestmark = pytest.mark.skipif(shutil.which("ffprobe") is None,
                                reason="ffprobe is needed to verify the audio")


@pytest.fixture(scope="module")
def gig(fixture_path, tmp_path_factory) -> dict:
    """Run ``fourfloor batch`` over a two-track folder and hand back the paths."""
    root = tmp_path_factory.mktemp("gig")
    originals, out = root / "originals", root / "friday"
    originals.mkdir()
    for name in NAMES:
        shutil.copy(fixture_path, originals / name)

    code = cli.main(["batch", str(originals), "-o", str(out),
                     "--bpm", str(TARGET_BPM), "--length", "1:00",
                     "--set", "Friday at Basement", "--quiet"])
    assert code == 0
    manifest = json.loads((out / "set.json").read_text(encoding="utf8"))
    return {"originals": originals, "out": out, "manifest": manifest}


def probe(path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:format_tags:stream=codec_name,channels",
         "-select_streams", "a:0", "-of", "json", str(path)],
        capture_output=True, check=True, timeout=120)
    return json.loads(proc.stdout.decode("utf8"))


# ---------------------------------------------------------------------------
# the renders
# ---------------------------------------------------------------------------

def test_both_tracks_rendered(gig) -> None:
    s = gig["manifest"]["summary"]
    assert (s["total"], s["ok"], s["failed"]) == (2, 2, 0)


def test_the_mp3s_are_real_audio_at_the_set_tempo(gig) -> None:
    for track in gig["manifest"]["tracks"]:
        info = probe(track["output"])
        assert info["streams"][0]["codec_name"] == "mp3"
        assert float(info["format"]["duration"]) > 60.0
        assert track["bpm"] == pytest.approx(TARGET_BPM)


def test_every_remix_kept_its_name(gig) -> None:
    assert [Path(t["output"]).name for t in gig["manifest"]["tracks"]] == [
        "01 Nuit Blanche — Édition.house.mp3", "02 second track.house.mp3"]


def test_each_remix_has_its_session_file_beside_it(gig) -> None:
    for track in gig["manifest"]["tracks"]:
        sess = export.session_path_for(Path(track["output"]))
        assert sess.is_file()
        assert json.loads(sess.read_text(encoding="utf8"))["bpm"] == TARGET_BPM


def test_the_set_has_the_structural_cues_a_dj_needs(gig) -> None:
    for track in gig["manifest"]["tracks"]:
        kinds = {c["kind"] for c in track["cues"]}
        assert {"intro", "build", "drop", "outro"} <= kinds


# ---------------------------------------------------------------------------
# the export that the batch triggered
# ---------------------------------------------------------------------------

def test_the_batch_wrote_the_dj_files(gig) -> None:
    assert (gig["out"] / "rekordbox.xml").is_file()
    assert (gig["out"] / "cues.csv").is_file()
    assert (gig["out"] / "set.json").is_file()


def test_the_collection_holds_both_tracks_under_the_set_name(gig) -> None:
    root = ET.fromstring((gig["out"] / "rekordbox.xml").read_text(encoding="utf8"))
    assert root.find("COLLECTION").get("Entries") == "2"
    node = root.find("PLAYLISTS/NODE/NODE")
    assert node.get("Name") == "Friday at Basement"
    assert node.get("Entries") == "2"


def test_every_location_points_at_a_file_that_exists(gig) -> None:
    """The one thing that makes an import fail silently and show red tracks."""
    root = ET.fromstring((gig["out"] / "rekordbox.xml").read_text(encoding="utf8"))
    for track in root.findall("COLLECTION/TRACK"):
        location = track.get("Location")
        assert location.startswith("file://localhost/")
        assert " " not in location
        assert Path(unquote(location[len("file://localhost"):])).is_file()


def test_the_grid_is_anchored_and_the_hot_cues_are_lettered(gig) -> None:
    root = ET.fromstring((gig["out"] / "rekordbox.xml").read_text(encoding="utf8"))
    for track in root.findall("COLLECTION/TRACK"):
        tempo = track.find("TEMPO")
        assert tempo.get("Bpm") == "124.00" and tempo.get("Metro") == "4/4"
        marks = track.findall("POSITION_MARK")
        assert [m for m in marks if m.get("Num") == "-1"], "no memory cue"
        hot = [m for m in marks if m.get("Num") != "-1"]
        assert [m.get("Num") for m in hot] == [str(i) for i in range(len(hot))]
        assert {m.get("Name") for m in hot} >= {"intro", "drop 1", "outro"}


def test_the_cue_sheet_covers_both_tracks(gig) -> None:
    import csv
    rows = list(csv.DictReader((gig["out"] / "cues.csv").read_text(encoding="utf8")
                               .splitlines()))
    assert len({r["file"] for r in rows}) == 2
    assert {r["hot_cue"] for r in rows} >= set("ABC")


# ---------------------------------------------------------------------------
# tagging the finished set, and resuming it
# ---------------------------------------------------------------------------

def test_tagging_the_finished_set_leaves_the_audio_alone(gig) -> None:
    mp3s = sorted(gig["out"].glob("*.house.mp3"))
    before = {p: probe(p)["format"]["duration"] for p in mp3s}

    code = cli.main(["export", str(gig["out"]), "--set", "Friday at Basement",
                     "--format", "tags,serato", "--out", str(gig["out"])])
    assert code == 0

    for path in mp3s:
        info = probe(path)
        assert info["format"]["duration"] == before[path]
        tags = {k.lower(): v for k, v in info["format"]["tags"].items()}
        assert tags["tbpm"] == "124.00"
        assert tags["initialkey"] == tags["tkey"]
        assert "drop 1" in tags["comment"]
        blob = export.read_id3(path)["GEOB"][export.SERATO_MARKERS2]
        cues = [e for e in export.parse_serato_markers2(blob) if e["type"] == "CUE"]
        assert [c["name"] for c in cues][:1] == ["intro"]


def test_resuming_the_same_batch_renders_nothing_again(gig) -> None:
    """Run last: it re-runs the command over the folder the other tests read."""
    code = cli.main(["batch", str(gig["originals"]), "-o", str(gig["out"]),
                     "--bpm", str(TARGET_BPM), "--length", "1:00",
                     "--set", "Friday at Basement", "--resume", "--quiet"])
    assert code == 0
    manifest = json.loads((gig["out"] / "set.json").read_text(encoding="utf8"))
    assert manifest["summary"]["skipped"] == 2
    assert manifest["summary"]["ok"] == 0
    assert manifest["summary"]["elapsed"] < 10.0
