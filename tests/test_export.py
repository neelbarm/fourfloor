"""The DJ handoff files: Rekordbox XML and the generic cue sheet.

Nothing here touches audio -- a session file and a placeholder on disk is all
the exporters need, which keeps the structural checks instant. The ID3 writing
that does touch audio lives in ``test_export_tags.py``.
"""

from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import pytest

from fourfloor import export
from fourfloor.export import ExportError

# The seven structural moments a club arrangement produces, plus the end
# marker the session file always appends.
CUE_KINDS = [
    ("intro", "intro", 0, 0.0),
    ("build 1", "build", 16, 30.9677),
    ("drop 1", "drop", 24, 46.4516),
    ("breakdown", "breakdown", 64, 123.871),
    ("build 2", "build", 80, 154.8387),
    ("drop 2", "drop", 88, 170.3226),
    ("outro", "outro", 120, 232.2581),
    ("end", "end", 136, 263.226),
]


def make_session(bpm: float = 124.0, key: str = "Dm", camelot: str = "7A",
                 first_downbeat: float = 0.0, cues=None, duration: float = 263.226,
                 file: str = "x.house.mp3") -> dict:
    """A session dict shaped like the one :mod:`fourfloor.session` writes."""
    cues = cues if cues is not None else [
        {"name": n, "kind": k, "bar": b, "time": t} for n, k, b, t in CUE_KINDS
    ]
    return {
        "schema": 2, "generator": "fourfloor", "file": file,
        "bpm": bpm, "first_downbeat_sec": first_downbeat, "beats_per_bar": 4,
        "key": key, "camelot": camelot, "semitone_shift": 0,
        "duration": duration, "bars": 136, "cues": cues, "sections": [],
        "energy_per_bar": [], "loudness": {"peak_db": -1.0, "rms_db": -8.4},
        "source": {"file": "x.mp3", "bpm": 80.0, "key": "Dm", "camelot": "7A"},
    }


def place(folder: Path, name: str, **session_kw) -> Path:
    """Write a placeholder mp3 plus its session file, and return the mp3 path."""
    folder.mkdir(parents=True, exist_ok=True)
    mp3 = folder / name
    mp3.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 2048)
    import json
    export.session_path_for(mp3).write_text(
        json.dumps(make_session(file=name, **session_kw)), encoding="utf8")
    return mp3


@pytest.fixture
def one_track(tmp_path) -> Path:
    return place(tmp_path / "set", "a song.house.mp3")


@pytest.fixture
def two_tracks(tmp_path) -> Path:
    """A folder with a plain name and a name full of the things that break URLs."""
    folder = tmp_path / "set"
    place(folder, "plain.house.mp3")
    place(folder, "Nuit Blanche — Édition #2 & co.house.mp3", key="Am", camelot="8A")
    return folder


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def test_session_path_replaces_only_the_audio_suffix() -> None:
    assert export.session_path_for(Path("a/lofi-7.house.mp3")).name \
        == "lofi-7.house.session.json"


def test_a_remix_without_a_session_file_is_a_clear_error(tmp_path) -> None:
    lonely = tmp_path / "orphan.house.mp3"
    lonely.write_bytes(b"\x00")
    with pytest.raises(ExportError, match="no session file"):
        export.collect(lonely)


def test_an_empty_folder_says_what_to_run(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ExportError, match="no remixes"):
        export.collect(tmp_path / "empty")


def test_collect_prefers_the_mp3_over_the_wav_of_the_same_remix(tmp_path) -> None:
    """A remix is written as both; a DJ loads the mp3, so only it should appear."""
    folder = tmp_path / "set"
    mp3 = place(folder, "song.house.mp3")
    wav = folder / "song.house.wav"
    wav.write_bytes(b"RIFF")
    tracks = export.collect(folder)
    assert [t.path for t in tracks] == [mp3]


def test_title_drops_the_house_marker_and_the_suffix_is_opt_in(tmp_path) -> None:
    place(tmp_path / "s", "Midnight Drive.house.mp3")
    assert export.collect(tmp_path / "s")[0].title == "Midnight Drive"
    suffixed = export.collect(tmp_path / "s", suffix=True)[0].title
    assert suffixed == "Midnight Drive (fourfloor house remix)"


def test_hot_cues_exclude_the_end_marker_and_cap_at_eight(tmp_path) -> None:
    many = [{"name": f"c{i}", "kind": "drop", "bar": i, "time": float(i)}
            for i in range(12)]
    many.append({"name": "end", "kind": "end", "bar": 99, "time": 400.0})
    place(tmp_path / "s", "many.house.mp3", cues=many)
    track = export.collect(tmp_path / "s")[0]
    assert len(track.cues) == 13
    assert len(track.hot_cues) == export.MAX_HOT_CUES
    assert all(c.kind != "end" for c in track.hot_cues)


def test_cues_come_back_in_time_order_however_the_session_stored_them(tmp_path) -> None:
    shuffled = [
        {"name": "drop 1", "kind": "drop", "bar": 24, "time": 46.0},
        {"name": "intro", "kind": "intro", "bar": 0, "time": 0.0},
        {"name": "outro", "kind": "outro", "bar": 120, "time": 232.0},
    ]
    place(tmp_path / "s", "x.house.mp3", cues=shuffled)
    times = [c.seconds for c in export.collect(tmp_path / "s")[0].cues]
    assert times == sorted(times)


# ---------------------------------------------------------------------------
# rekordbox xml
# ---------------------------------------------------------------------------

def parse(folder: Path, set_name: str = "Friday") -> ET.Element:
    tracks = export.collect(folder)
    return ET.fromstring(export.rekordbox_xml(tracks, set_name))


def test_the_document_is_well_formed_dj_playlists(two_tracks) -> None:
    root = parse(two_tracks)
    assert root.tag == "DJ_PLAYLISTS"
    assert root.get("Version") == "1.0.0"
    assert root.find("PRODUCT") is not None
    assert root.find("COLLECTION") is not None
    assert root.find("PLAYLISTS") is not None


def test_the_file_starts_with_the_xml_declaration(two_tracks) -> None:
    text = export.rekordbox_xml(export.collect(two_tracks), "Friday")
    assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>')


def test_collection_entries_counts_the_tracks(two_tracks) -> None:
    """Rekordbox trusts Entries; a wrong count truncates the import."""
    root = parse(two_tracks)
    collection = root.find("COLLECTION")
    assert collection.get("Entries") == "2"
    assert len(collection.findall("TRACK")) == 2


def test_track_carries_the_fields_a_deck_needs(one_track) -> None:
    track = parse(one_track.parent).find("COLLECTION/TRACK")
    assert track.get("Name") == "a song"
    assert track.get("Artist") == "fourfloor"
    assert track.get("TotalTime") == "263"          # whole seconds, as rekordbox wants
    assert track.get("AverageBpm") == "124.00"
    assert track.get("Tonality") == "Dm"
    assert track.get("Kind") == "MP3 File"
    assert "drop 1" in track.get("Comments")


def test_tempo_anchors_the_grid_at_the_first_downbeat(tmp_path) -> None:
    place(tmp_path / "s", "x.house.mp3", first_downbeat=1.234, bpm=126.5)
    tempo = parse(tmp_path / "s").find("COLLECTION/TRACK/TEMPO")
    assert tempo.get("Inizio") == "1.234"
    assert tempo.get("Bpm") == "126.50"
    assert tempo.get("Metro") == "4/4"
    assert tempo.get("Battito") == "1"


def test_there_is_exactly_one_tempo_marker(one_track) -> None:
    """The remix sits on an exact grid, so a second anchor could only drift."""
    assert len(parse(one_track.parent).findall("COLLECTION/TRACK/TEMPO")) == 1


def test_hot_cues_are_a_to_h_in_order(one_track) -> None:
    marks = parse(one_track.parent).findall("COLLECTION/TRACK/POSITION_MARK")
    hot = [m for m in marks if m.get("Num") != "-1"]
    assert [m.get("Num") for m in hot] == ["0", "1", "2", "3", "4", "5", "6"]
    assert [m.get("Name") for m in hot] == [
        "intro", "build 1", "drop 1", "breakdown", "build 2", "drop 2", "outro"]
    assert all(m.get("Type") == "0" for m in hot)
    assert [m.get("Start") for m in hot][:3] == ["0.000", "30.968", "46.452"]
    assert all(m.get("Red") and m.get("Green") and m.get("Blue") for m in hot)


def test_the_end_marker_does_not_take_a_hot_cue_button(one_track) -> None:
    names = [m.get("Name")
             for m in parse(one_track.parent).findall("COLLECTION/TRACK/POSITION_MARK")]
    assert "end" not in names


def test_a_memory_cue_sits_on_the_first_downbeat(tmp_path) -> None:
    """Num="-1" is how rekordbox marks a memory cue rather than a hot cue."""
    place(tmp_path / "s", "x.house.mp3", first_downbeat=0.5)
    marks = parse(tmp_path / "s").findall("COLLECTION/TRACK/POSITION_MARK")
    memory = [m for m in marks if m.get("Num") == "-1"]
    assert len(memory) == 1
    assert memory[0].get("Start") == "0.500"
    assert memory[0].get("Type") == "0"


def test_location_is_a_percent_encoded_file_url(two_tracks) -> None:
    """Spaces and unicode have to survive the round trip or every track imports red."""
    locations = [t.get("Location")
                 for t in parse(two_tracks).findall("COLLECTION/TRACK")]
    tricky = [loc for loc in locations if "Nuit" in unquote(loc)][0]
    assert tricky.startswith("file://localhost/")
    assert " " not in tricky
    assert "%20" in tricky
    assert "É" not in tricky and "%C3%89" in tricky          # utf-8 percent bytes
    assert "&" not in tricky and "#" not in tricky
    path = Path(unquote(tricky[len("file://localhost"):]))
    assert path.is_file()
    assert path.name == "Nuit Blanche — Édition #2 & co.house.mp3"


def test_location_is_absolute_even_for_a_relative_target(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    place(Path("set"), "x.house.mp3")
    loc = parse(Path("set")).find("COLLECTION/TRACK").get("Location")
    assert unquote(loc[len("file://localhost"):]).startswith("/")


def test_the_playlist_node_is_named_after_the_set(two_tracks) -> None:
    root = parse(two_tracks, "Friday at Basement")
    node = root.find("PLAYLISTS/NODE/NODE")
    assert node.get("Name") == "Friday at Basement"
    assert node.get("Type") == "1"                  # 1 = playlist, 0 = folder
    assert node.get("KeyType") == "0"               # keys are TrackIDs
    assert node.get("Entries") == "2"
    assert [t.get("Key") for t in node.findall("TRACK")] == ["1", "2"]


def test_playlist_keys_match_the_collection_track_ids(two_tracks) -> None:
    root = parse(two_tracks)
    ids = {t.get("TrackID") for t in root.findall("COLLECTION/TRACK")}
    keys = {t.get("Key") for t in root.findall("PLAYLISTS/NODE/NODE/TRACK")}
    assert keys == ids


def test_write_rekordbox_lands_in_the_out_folder(two_tracks, tmp_path) -> None:
    out = tmp_path / "deliver"
    path = export.write_rekordbox(export.collect(two_tracks), out, "Friday")
    assert path == out / "rekordbox.xml"
    assert ET.fromstring(path.read_text(encoding="utf8")).tag == "DJ_PLAYLISTS"


# ---------------------------------------------------------------------------
# cues.csv
# ---------------------------------------------------------------------------

def test_csv_has_a_row_per_cue_with_name_seconds_and_bar(one_track) -> None:
    rows = list(csv.reader(io.StringIO(export.cues_csv(export.collect(one_track.parent)))))
    assert rows[0][:4] == ["file", "cue", "seconds", "bar"]
    body = rows[1:]
    assert len(body) == len(CUE_KINDS)
    assert [r[1] for r in body] == [c[0] for c in CUE_KINDS]
    assert [r[3] for r in body] == [str(c[2]) for c in CUE_KINDS]
    # the sheet is written to the millisecond, which is finer than a DJ can cue
    assert [float(r[2]) for r in body] == pytest.approx([c[3] for c in CUE_KINDS],
                                                        abs=1e-3)
    assert all(r[0] == "a song.house.mp3" for r in body)


def test_csv_letters_the_hot_cues_and_leaves_the_end_blank(one_track) -> None:
    rows = list(csv.DictReader(io.StringIO(
        export.cues_csv(export.collect(one_track.parent)))))
    assert [r["hot_cue"] for r in rows] == list("ABCDEFG") + [""]
    assert rows[-1]["kind"] == "end"


def test_csv_holds_every_track_in_one_sheet(two_tracks) -> None:
    rows = list(csv.DictReader(io.StringIO(
        export.cues_csv(export.collect(two_tracks)))))
    assert len({r["file"] for r in rows}) == 2


def test_csv_quotes_a_filename_with_a_comma(tmp_path) -> None:
    place(tmp_path / "s", "hey, listen.house.mp3")
    text = export.cues_csv(export.collect(tmp_path / "s"))
    assert '"hey, listen.house.mp3"' in text
    assert list(csv.DictReader(io.StringIO(text)))[0]["file"] == "hey, listen.house.mp3"


# ---------------------------------------------------------------------------
# format selection
# ---------------------------------------------------------------------------

def test_no_format_means_the_two_that_cannot_damage_anything() -> None:
    assert export.normalise_formats(None) == ["rekordbox", "csv"]


def test_formats_can_be_repeated_or_comma_separated() -> None:
    assert export.normalise_formats(["csv,rekordbox", "csv"]) == ["csv", "rekordbox"]


def test_all_expands_to_every_format() -> None:
    assert export.normalise_formats(["all"]) == list(export.FORMATS)


def test_an_unknown_format_lists_the_real_ones() -> None:
    with pytest.raises(ExportError, match="traktor"):
        export.normalise_formats(["traktor"])


def test_export_writes_both_files_and_touches_no_audio(two_tracks, tmp_path) -> None:
    before = {p: p.read_bytes() for p in two_tracks.glob("*.mp3")}
    out = tmp_path / "deliver"
    res = export.export(two_tracks, "Friday", formats=["rekordbox", "csv"], out_dir=out)
    assert (out / "rekordbox.xml").is_file()
    assert (out / "cues.csv").is_file()
    assert res.tagged == []
    assert {p: p.read_bytes() for p in two_tracks.glob("*.mp3")} == before


def test_export_of_a_single_file_defaults_to_its_own_folder(one_track) -> None:
    res = export.export(one_track, "Friday")
    assert res.files["rekordbox"].parent == one_track.parent
    assert len(res.tracks) == 1


def test_export_to_dict_is_json_shaped(two_tracks, tmp_path) -> None:
    import json
    res = export.export(two_tracks, "Friday", out_dir=tmp_path / "d")
    data = json.loads(json.dumps(res.to_dict()))
    assert data["set"] == "Friday"
    assert len(data["tracks"]) == 2
    assert data["tracks"][0]["hot_cues"][0]["letter"] == "A"
