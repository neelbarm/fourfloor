"""ID3v2.3 tagging and Serato cue blobs, on a real mp3.

The load-bearing promise of this module is that tagging cannot damage a remix:
only the tag at the head of the file is replaced and every MPEG frame after it
is copied through untouched. Each test that writes a tag checks that with
ffprobe as well as byte for byte.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from fourfloor import export
from fourfloor.export import ExportError

from test_export import make_session

pytestmark = pytest.mark.skipif(shutil.which("ffprobe") is None,
                                reason="ffprobe is needed to verify the audio")


def ffprobe_tags(path) -> dict:
    """Every format-level tag ffprobe can see, lower-cased keys."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags",
         "-of", "json", str(path)],
        capture_output=True, check=True, timeout=60)
    import json
    tags = json.loads(proc.stdout.decode("utf8")).get("format", {}).get("tags", {})
    return {k.lower(): v for k, v in tags.items()}


@pytest.fixture
def remix(fixture_path, tmp_path) -> export.Track:
    """A copy of the bundled mp3 with a session file beside it, safe to tag."""
    import json

    mp3 = tmp_path / "Nuit Blanche — Édition.house.mp3"
    shutil.copy(fixture_path, mp3)
    export.session_path_for(mp3).write_text(
        json.dumps(make_session(file=mp3.name)), encoding="utf8")
    return export.collect(mp3)[0]


# ---------------------------------------------------------------------------
# the audio survives
# ---------------------------------------------------------------------------

def test_tagging_leaves_the_audio_stream_identical(remix) -> None:
    before = export.probe(remix.path)
    export.tag_file(remix)
    assert export.probe(remix.path) == before


def test_tagging_copies_every_mpeg_frame_byte_for_byte(remix) -> None:
    """The strongest form of the promise: strip the tags and compare the rest."""
    audio_before = export.strip_id3(remix.path.read_bytes())
    export.tag_file(remix)
    assert export.strip_id3(remix.path.read_bytes()) == audio_before


def test_tagging_twice_replaces_the_tag_rather_than_stacking_them(remix) -> None:
    export.tag_file(remix)
    size_once = remix.path.stat().st_size
    export.tag_file(remix)
    assert remix.path.stat().st_size == size_once
    raw = remix.path.read_bytes()
    assert raw.count(b"ID3\x03\x00") == 1


def test_tagging_a_wav_is_refused_rather_than_attempted(remix, tmp_path) -> None:
    wav = tmp_path / "x.house.wav"
    wav.write_bytes(b"RIFF")
    remix.path = wav
    with pytest.raises(ExportError, match="only tag mp3"):
        export.tag_file(remix)


def test_no_temp_file_is_left_behind(remix) -> None:
    export.tag_file(remix)
    assert not list(remix.path.parent.glob("*.fourfloor-tmp"))


# ---------------------------------------------------------------------------
# what ffprobe reads back
# ---------------------------------------------------------------------------

def test_the_standard_frames_round_trip_through_ffprobe(remix) -> None:
    export.tag_file(remix)
    tags = ffprobe_tags(remix.path)
    assert tags["title"] == "Nuit Blanche — Édition"
    assert tags["artist"] == "fourfloor"
    assert tags["tbpm"] == "124.00"
    assert tags["tkey"] == "Dm"
    assert tags["genre"] == "House"
    assert "drop 1" in tags["comment"]


def test_the_rekordbox_friendly_txxx_frames_round_trip(remix) -> None:
    """Both apps read INITIALKEY from TXXX; several versions ignore TKEY."""
    export.tag_file(remix)
    tags = ffprobe_tags(remix.path)
    assert tags["initialkey"] == "Dm"
    assert tags["camelot"] == "7A"


def test_the_title_suffix_is_written_when_asked(fixture_path, tmp_path) -> None:
    import json
    mp3 = tmp_path / "Song.house.mp3"
    shutil.copy(fixture_path, mp3)
    export.session_path_for(mp3).write_text(json.dumps(make_session()), encoding="utf8")
    export.tag_file(export.collect(mp3, suffix=True)[0])
    assert ffprobe_tags(mp3)["title"] == "Song (fourfloor house remix)"


def test_the_comment_carries_the_whole_cue_list(remix) -> None:
    export.tag_file(remix)
    comment = ffprobe_tags(remix.path)["comment"]
    for letter, name in zip("ABCDEFG", ["intro", "build 1", "drop 1", "breakdown",
                                        "build 2", "drop 2", "outro"]):
        assert f"{letter} {name}" in comment


# ---------------------------------------------------------------------------
# the frame writer itself
# ---------------------------------------------------------------------------

def test_latin1_is_used_while_it_fits_and_utf16_takes_over_when_it_does_not() -> None:
    """ISO-8859-1 is what every reader handles; UTF-16 is the only alternative
    ID3v2.3 allows, so an em dash costs two bytes a character and nothing else."""
    assert export.text_frame("TIT2", "plain")[10] == 0x00
    assert export.text_frame("TIT2", "Édition")[10] == 0x00      # É fits in latin-1
    frame = export.text_frame("TIT2", "Nuit — Blanche")          # the em dash does not
    assert frame[10] == 0x01
    assert frame[11:13] == b"\xff\xfe"               # the byte order mark


def test_read_id3_round_trips_everything_the_writer_writes(tmp_path) -> None:
    path = tmp_path / "t.mp3"
    path.write_bytes(export.id3v23_tag([
        export.text_frame("TIT2", "Édition"),
        export.text_frame("TBPM", "126.00"),
        export.txxx_frame("INITIALKEY", "Am"),
        export.comment_frame("hello"),
        export.geob_frame("blob", b"\x00\x01\x02"),
    ]) + b"\xff\xfb\x90\x00")
    got = export.read_id3(path)
    assert got["TIT2"] == "Édition"
    assert got["TBPM"] == "126.00"
    assert got["TXXX"]["INITIALKEY"] == "Am"
    assert got["COMM"][""] == "hello"
    assert got["GEOB"]["blob"] == b"\x00\x01\x02"


def test_read_id3_on_an_untagged_file_is_empty_not_an_error(tmp_path) -> None:
    path = tmp_path / "bare.mp3"
    path.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 64)
    assert export.read_id3(path)["TXXX"] == {}


def test_strip_id3_removes_only_the_leading_tag(tmp_path) -> None:
    audio = b"\xff\xfb" + b"\x11" * 500
    assert export.strip_id3(export.id3v23_tag([], padding=8) + audio) == audio
    assert export.strip_id3(audio) == audio


# ---------------------------------------------------------------------------
# serato
# ---------------------------------------------------------------------------

def test_serato_markers2_round_trips_through_the_tagged_file(remix) -> None:
    export.tag_file(remix, serato=True)
    blob = export.read_id3(remix.path)["GEOB"][export.SERATO_MARKERS2]
    entries = export.parse_serato_markers2(blob)
    cues = [e for e in entries if e["type"] == "CUE"]
    assert [c["index"] for c in cues] == [0, 1, 2, 3, 4, 5, 6]
    assert [c["name"] for c in cues] == ["intro", "build 1", "drop 1", "breakdown",
                                         "build 2", "drop 2", "outro"]
    assert cues[2]["millis"] == 46452                 # drop 1 at 46.4516 s
    assert cues[2]["color"] == export.CUE_COLORS["drop"]


def test_serato_object_is_a_version_header_then_wrapped_base64() -> None:
    blob = export.serato_markers2(
        [export.Cue("drop 1", 46.4516, 24, "drop")])
    assert blob.startswith(b"\x01\x01")
    assert blob.endswith(b"\x00")
    body = blob[2:-1]
    assert all(len(line) <= 72 for line in body.split(b"\n"))
    assert set(body.replace(b"\n", b"")) <= set(
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")


def test_serato_payload_holds_a_colour_and_a_bpm_lock() -> None:
    entries = export.parse_serato_markers2(export.serato_markers2([], bpm_lock=True))
    kinds = {e["type"] for e in entries}
    assert {"COLOR", "BPMLOCK"} <= kinds
    assert [e for e in entries if e["type"] == "BPMLOCK"][0]["locked"] is True


def test_serato_cue_labels_survive_unicode() -> None:
    blob = export.serato_markers2([export.Cue("brèak ↓", 12.0, 8, "breakdown")])
    cue = [e for e in export.parse_serato_markers2(blob) if e["type"] == "CUE"][0]
    assert cue["name"] == "brèak ↓"


def test_serato_refuses_a_blob_that_is_not_markers2() -> None:
    with pytest.raises(ExportError, match="version header"):
        export.parse_serato_markers2(b"nope")


def test_serato_cues_stop_at_eight() -> None:
    cues = [export.Cue(f"c{i}", float(i), i, "drop") for i in range(12)]
    entries = export.parse_serato_markers2(export.serato_markers2(cues))
    assert len([e for e in entries if e["type"] == "CUE"]) == export.MAX_HOT_CUES


def test_export_with_serato_says_the_format_is_unverified(remix, tmp_path) -> None:
    res = export.export(remix.path, "Friday", formats=["serato"], out_dir=tmp_path)
    assert any("Serato" in n and "check one track" in n for n in res.notes)
    assert res.tagged and res.tagged[0]["serato"] is True
