"""The listening-feedback store: markers, ratings, votes, and the digest.

These tests work on a hand-built remix folder rather than a rendered one --
a session file and a plan file are all :mod:`fourfloor.feedback` reads, and
building them by hand keeps the whole file fast and makes the bar arithmetic
something you can check by eye.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from fourfloor import feedback, store

BAR = 60.0 / 124.0 * 4          # 1.935…s per bar at 124 BPM


@pytest.fixture
def remix(tmp_path: Path):
    """A library with one remix whose session and plan files are real enough."""
    lib = store.Library(tmp_path)
    rid, d = lib.create_remix()
    (d / "remix.mp3").write_bytes(b"\xff\xfb" + b"\x00" * 64)
    lib.save_remix_meta(rid, {
        "title": "Lofi 7", "bpm": 124.0, "key": "F minor", "camelot": "4A",
        "form": "club", "length": "2:00", "source_id": None,
    })
    (d / "remix.session.json").write_text(json.dumps({
        "schema": 2, "generator": "fourfloor", "file": "remix.mp3", "bpm": 124.0,
        "first_downbeat_sec": 0.0, "beats_per_bar": 4,
        "beat_duration_sec": round(60.0 / 124.0, 6),
        "key": "F minor", "camelot": "4A", "semitone_shift": 0,
        "duration": round(64 * BAR, 3), "bars": 64, "sections": [],
        "cues": [{"name": "intro", "bar": 0, "time": 0.0, "kind": "intro"},
                 {"name": "drop 1", "bar": 24, "time": round(24 * BAR, 4),
                  "kind": "drop"}],
        "energy_per_bar": [0.5] * 64, "loudness": {"peak_db": -0.3}, "source": {},
    }, indent=2), encoding="utf8")
    (d / "remix.plan.json").write_text(json.dumps({
        "target_bpm": 124.0, "total_bars": 64, "form": "club",
        "slots": [
            {"kind": "intro", "start_bar": 0, "bars": 16, "note": "filtered in"},
            {"kind": "build", "start_bar": 16, "bars": 8, "note": "riser"},
            {"kind": "drop", "start_bar": 24, "bars": 24, "note": "hook, full kit"},
            {"kind": "outro", "start_bar": 48, "bars": 16, "note": "filtered out"},
        ],
    }, indent=2), encoding="utf8")
    return lib, rid, d


# ---------------------------------------------------------------------------
# markers
# ---------------------------------------------------------------------------

def test_an_empty_remix_reads_as_an_empty_document(remix) -> None:
    _, rid, d = remix
    data = feedback.read(d, rid)
    assert data == {"schema": 1, "remix": rid, "markers": [], "ratings": [],
                    "votes": []}
    assert not (d / "feedback.json").exists(), "reading should not write"


def test_a_marker_lands_on_the_bar_and_slot_the_arranger_built(remix) -> None:
    _, rid, d = remix
    m = feedback.add_marker(d, 25.5 * BAR, "drums-fake", "hats sound plastic", rid)
    assert m["bar"] == 26                       # 0-based bar 25 → 1-based 26
    assert m["slot"] == "drop"
    assert m["slot_bars"] == "drop bars 25-48"
    assert m["note"] == "hats sound plastic"
    assert m["at"] > 0
    assert feedback.read(d, rid)["markers"] == [m]


def test_markers_append_and_never_overwrite(remix) -> None:
    _, rid, d = remix
    for i, cat in enumerate(("off-beat", "vocal-buried", "good")):
        feedback.add_marker(d, i * 8 * BAR, cat, f"note {i}", rid)
    marks = feedback.read(d, rid)["markers"]
    assert [m["category"] for m in marks] == ["off-beat", "vocal-buried", "good"]
    assert [m["bar"] for m in marks] == [1, 9, 17]


def test_a_marker_is_mirrored_into_the_session_file(remix) -> None:
    """The DJ handoff has to carry the notes; it is what leaves the machine."""
    _, rid, d = remix
    feedback.add_marker(d, 4.0, "clash", "wrong bass note", rid)
    feedback.add_marker(d, 8.0, "too-loud", "", rid)
    session = json.loads((d / "remix.session.json").read_text())
    assert [f["category"] for f in session["feedback"]] == ["clash", "too-loud"]
    assert session["feedback"][0]["note"] == "wrong bass note"
    assert "at" in session["feedback"][0] and "bar" in session["feedback"][0]

    from fourfloor import session as session_mod
    assert session_mod.validate(session) == [], "the field must stay additive"


def test_a_marker_past_the_end_is_pulled_back_to_the_end(remix) -> None:
    _, rid, d = remix
    m = feedback.add_marker(d, 99999.0, "boring", "", rid)
    assert m["time"] == pytest.approx(round(64 * BAR, 3))


@pytest.mark.parametrize("bad", ["", "wrong", "OFF BEAT", None, 7])
def test_only_the_known_categories_are_stored(remix, bad) -> None:
    _, rid, d = remix
    with pytest.raises(feedback.FeedbackError):
        feedback.add_marker(d, 1.0, bad, "", rid)


@pytest.mark.parametrize("bad", ["banana", None, -3, float("nan"), float("inf")])
def test_a_marker_needs_a_real_time(remix, bad) -> None:
    _, rid, d = remix
    with pytest.raises(feedback.FeedbackError):
        feedback.add_marker(d, bad, "good", "", rid)


def test_a_note_is_reduced_to_one_short_safe_line(remix) -> None:
    _, rid, d = remix
    m = feedback.add_marker(d, 1.0, "good", "a\x00b\nc\td   e" + "x" * 900, rid)
    assert m["note"].startswith("ab c d e")
    assert len(m["note"]) <= feedback.MAX_NOTE
    assert "\n" not in m["note"] and "\x00" not in m["note"]


def test_a_marker_survives_a_remix_with_no_plan(tmp_path: Path) -> None:
    lib = store.Library(tmp_path)
    rid, d = lib.create_remix()
    m = feedback.add_marker(d, 12.0, "off-beat", "", rid)
    assert m["bar"] == 0 and m["slot"] == ""      # unknown grid, still recorded
    assert feedback.read(d, rid)["markers"] == [m]


# ---------------------------------------------------------------------------
# ratings and votes
# ---------------------------------------------------------------------------

def test_the_rating_that_stands_is_the_last_of_each_field(remix) -> None:
    _, rid, d = remix
    feedback.add_rating(d, 2, "drums are fake", rid)
    feedback.add_rating(d, 4, None, rid)          # fixed the drums, re-rated
    data = feedback.read(d, rid)
    assert len(data["ratings"]) == 2, "append-only: the 2 is still on record"
    assert feedback.latest_rating(data) == {"stars": 4,
                                            "verdict": "drums are fake",
                                            "at": pytest.approx(data["ratings"][-1]["at"])}


@pytest.mark.parametrize("bad", [0, 6, -1, "many", 2.7e9])
def test_a_rating_runs_from_one_to_five(remix, bad) -> None:
    _, rid, d = remix
    with pytest.raises(feedback.FeedbackError):
        feedback.add_rating(d, bad, None, rid)


def test_an_empty_rating_is_refused(remix) -> None:
    _, rid, d = remix
    with pytest.raises(feedback.FeedbackError):
        feedback.add_rating(d, None, "   ", rid)


def test_a_vote_records_which_side_won_and_why(remix) -> None:
    _, rid, d = remix
    other = store.new_id()
    row = feedback.add_vote(d, other, "b", "more space in the drop",
                            label="this one", other_label="take 2", rid=rid)
    assert row["other"] == other and row["prefer"] == "b" and row["winner"] == "other"
    assert row["reason"] == "more space in the drop"
    assert feedback.read(d, rid)["votes"] == [row]


@pytest.mark.parametrize("bad", ["", "c", "A/B", None])
def test_a_vote_is_for_a_or_b(remix, bad) -> None:
    _, rid, d = remix
    with pytest.raises(feedback.FeedbackError):
        feedback.add_vote(d, "x", bad, rid=rid)


# ---------------------------------------------------------------------------
# apply(): one POST body
# ---------------------------------------------------------------------------

def test_apply_takes_a_marker_a_rating_and_a_vote_at_once(remix) -> None:
    _, rid, d = remix
    out = feedback.apply(d, {
        "marker": {"time": 30 * BAR, "category": "transition", "note": "lurches"},
        "stars": 3, "verdict": "close",
        "vote": {"other": "source", "prefer": "a", "reason": "ours grooves"},
    }, rid)
    assert len(out["markers"]) == 1 and out["markers"][0]["slot"] == "drop"
    assert feedback.latest_rating(out) == {"stars": 3, "verdict": "close",
                                           "at": pytest.approx(out["ratings"][-1]["at"])}
    assert out["votes"][0]["winner"] == "this"


@pytest.mark.parametrize("payload", [{}, {"nope": 1}, "a string", None])
def test_apply_refuses_a_body_with_nothing_in_it(remix, payload) -> None:
    _, rid, d = remix
    with pytest.raises(feedback.FeedbackError):
        feedback.apply(d, payload, rid)


def test_a_broken_feedback_file_does_not_take_the_app_down(remix) -> None:
    _, rid, d = remix
    (d / "feedback.json").write_text("{not json", encoding="utf8")
    assert feedback.read(d, rid)["markers"] == []
    feedback.add_marker(d, 1.0, "good", "", rid)   # and writing repairs it
    assert len(feedback.read(d, rid)["markers"]) == 1


# ---------------------------------------------------------------------------
# the digest
# ---------------------------------------------------------------------------

def test_the_digest_says_so_when_nobody_has_listened(tmp_path: Path) -> None:
    out = feedback.digest(store.Library(tmp_path))
    assert "No listening notes yet" in out
    assert "press M" in out


def test_the_digest_groups_by_category_with_bars_and_slots(remix) -> None:
    lib, rid, d = remix
    feedback.add_marker(d, 26 * BAR, "drums-fake", "plastic hats", rid)
    feedback.add_marker(d, 30 * BAR, "drums-fake", "no swing", rid)
    feedback.add_marker(d, 17 * BAR, "off-beat", "drags", rid)
    feedback.add_rating(d, 3, "nearly", rid)
    other_id = store.new_id()
    feedback.add_vote(d, other_id, "b", "take 2 breathes",
                      other_label="take 2", rid=rid)

    out = feedback.digest(lib)
    assert "1 remix with feedback" in out
    assert "Lofi 7" in out and rid in out
    assert "★★★☆☆" in out and "nearly" in out
    # grouped, with the bar, the slot and the arranger's own note for that slot
    assert "Drums fake  (2)" in out
    assert "bar  27" in out and "[drop bars 25-48]" in out
    assert "plastic hats" in out and "no swing" in out
    assert "the arranger said: hook, full kit" in out
    assert "Off-beat  (1)" in out and "[build bars 17-24]" in out
    # the other side is named *and* identified: two takes of one song share a
    # title, so "preferred take 2" on its own would not say which file
    assert f"A/B vs take 2 ({other_id}): preferred the other one" in out
    assert "take 2 breathes" in out
    # the categories come out in the order the chips are shown, not by count
    assert out.index("Off-beat") < out.index("Drums fake")


def test_the_digest_skips_remixes_nobody_marked(remix) -> None:
    lib, rid, d = remix
    quiet, _ = lib.create_remix()
    (lib.remix_dir(quiet) / "remix.mp3").write_bytes(b"x")
    lib.save_remix_meta(quiet, {"title": "Untouched", "bpm": 124.0})
    feedback.add_marker(d, 1.0, "good", "", rid)
    out = feedback.digest(lib)
    assert "Lofi 7" in out and "Untouched" not in out
    assert "1 remix with feedback" in out


def test_the_digest_can_be_narrowed_to_one_remix(remix) -> None:
    lib, rid, d = remix
    other, od = lib.create_remix()
    (od / "remix.mp3").write_bytes(b"x")
    lib.save_remix_meta(other, {"title": "Take 2", "bpm": 126.0})
    feedback.add_marker(d, 1.0, "good", "", rid)
    feedback.add_marker(od, 1.0, "boring", "", other)
    assert "Take 2" not in feedback.digest(lib, rid)
    assert "Lofi 7" not in feedback.digest(lib, other)


def test_the_module_prints_its_digest(remix, capsys) -> None:
    lib, rid, d = remix
    feedback.add_marker(d, 26 * BAR, "clash", "bass fights the vocal", rid)
    assert feedback.main(["--home", str(lib.home)]) == 0
    assert "bass fights the vocal" in capsys.readouterr().out

    assert feedback.main(["--home", str(lib.home), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["remixes"][0]["id"] == rid
    assert data["remixes"][0]["feedback"]["markers"][0]["category"] == "clash"
    assert [c["code"] for c in data["categories"]] == list(feedback.CATEGORIES)


# ---------------------------------------------------------------------------
# two writers, one file: the critic appends Gemini's issues here too
# ---------------------------------------------------------------------------

def gemini_rows(d: Path, rows: list[dict]) -> list[dict]:
    """Write markers the way ``fourfloor/critic/gemini.py`` writes them.

    Deliberately its shape and not ours: ``ts`` rather than ``at``, bars
    counted from zero, and a category this module has never heard of.
    """
    doc = json.loads((d / "feedback.json").read_text()) \
        if (d / "feedback.json").exists() else {"markers": []}
    keep = [m for m in doc.get("markers", []) if m.get("author") != "gemini"]
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mine = [{**r, "author": "gemini", "ts": stamp} for r in rows]
    doc["markers"] = sorted(keep + mine, key=lambda m: (m.get("time") or 0.0))
    (d / "feedback.json").write_text(json.dumps(doc, indent=2) + "\n")
    return mine


def test_a_marker_from_the_app_says_who_left_it(remix) -> None:
    _, rid, d = remix
    m = feedback.add_marker(d, 10.0, "good", "", rid)
    assert m["author"] == "neel"
    session = json.loads((d / "remix.session.json").read_text())
    assert session["feedback"][-1]["author"] == "neel"


def test_gemini_markers_are_read_beside_neels(remix) -> None:
    _, rid, d = remix
    feedback.add_marker(d, 26 * BAR, "drums-fake", "plastic hats", rid)
    gemini_rows(d, [
        {"time": round(18 * BAR, 2), "bar": 18, "category": "artifact",
         "note": "phase-vocoder ringing on the riser tail"},
    ])
    data = feedback.read(d, rid)
    assert [m["author"] for m in data["markers"]] == ["gemini", "neel"]
    heard = data["markers"][0]
    assert heard["category"] == "artifact"        # not one of ours, kept anyway
    assert heard["at"] > 0, "the ISO ts is read as a timestamp"
    assert heard["slot"] == "" and heard["note"].startswith("phase-vocoder")


def test_a_bar_the_critic_could_not_work_out_does_not_crash_anything(remix) -> None:
    lib, rid, d = remix
    gemini_rows(d, [{"time": 9.0, "bar": None, "category": "artifact",
                     "note": "no grid here"}])
    assert feedback.read(d, rid)["markers"][0]["bar"] == 0
    assert "no grid here" in feedback.digest(lib)


def test_both_writers_land_on_the_same_bar_numbers(remix) -> None:
    """The critic counts bars from zero and this module counts from one.

    Neither side is wrong and neither is going to change, so the bar is
    recomputed from the time -- the one thing they cannot disagree about.
    """
    lib, rid, d = remix
    t = 26 * BAR + 0.4
    feedback.add_marker(d, t, "drums-fake", "mine", rid)
    gemini_rows(d, [{"time": round(t, 2), "bar": 26,      # zero-based: same bar
                     "category": "drums-fake", "note": "theirs"}])
    marks = feedback.collect(lib)[0]["feedback"]["markers"]
    assert {m["bar"] for m in marks} == {27}, marks
    assert {m["slot"] for m in marks} == {"drop"}
    out = feedback.digest(lib)
    assert out.count("bar  27") == 2


def test_the_digest_names_the_second_listener_and_not_the_first(remix) -> None:
    lib, rid, d = remix
    feedback.add_marker(d, 26 * BAR, "drums-fake", "plastic hats", rid)
    gemini_rows(d, [{"time": round(30 * BAR, 2), "bar": 30,
                     "category": "artifact", "note": "smearing"}])
    out = feedback.digest(lib)
    assert "heard by  Gemini, Neel" in out
    assert "smearing  (Gemini)" in out
    assert "plastic hats  (Neel)" not in out, "the default voice is not annotated"
    # a category the critic invented gets its own group, tidied the same way
    # the chips are, so the block does not read in two house styles
    assert "Artifact  (1)" in out
    assert feedback.category_label("artifact") == "Artifact"
    assert feedback.category_label("off-beat") == "Off-beat"


def test_an_app_write_returns_gemini_rows_to_disk_untouched(remix) -> None:
    """Two writers on one file only works if neither tidies the other's rows."""
    _, rid, d = remix
    feedback.add_marker(d, 5.0, "good", "", rid)
    gemini_rows(d, [{"time": 9.0, "bar": None, "category": "artifact", "note": "x"},
                    {"time": 12.0, "bar": 6, "category": "clash", "note": None}])
    before = [m for m in json.loads((d / "feedback.json").read_text())["markers"]
              if m.get("author") == "gemini"]

    feedback.add_marker(d, 40.0, "boring", "same eight bars", rid)
    feedback.add_rating(d, 2, "not there yet", rid)
    feedback.add_vote(d, store.new_id(), "b", rid=rid)

    after = [m for m in json.loads((d / "feedback.json").read_text())["markers"]
             if m.get("author") == "gemini"]
    assert after == before, "we rewrote somebody else's records"
    assert len(after) == 2


def test_the_session_handback_carries_what_gemini_heard_too(remix) -> None:
    """The session file is what leaves the machine; it should carry both.

    The critic writes into ``feedback.json`` and knows nothing about the
    session file, so the array is rebuilt from every marker on the next app
    write rather than appended to one at a time.
    """
    _, rid, d = remix
    feedback.add_marker(d, 26 * BAR, "drums-fake", "plastic hats", rid)
    gemini_rows(d, [{"time": round(18 * BAR, 2), "bar": 18,
                     "category": "artifact", "note": "ringing"}])

    session = json.loads((d / "remix.session.json").read_text())
    assert {f["author"] for f in session["feedback"]} == {"neel"}, "not yet"

    feedback.add_marker(d, 50 * BAR, "good", "outro is lovely", rid)
    session = json.loads((d / "remix.session.json").read_text())
    assert {f["author"] for f in session["feedback"]} == {"neel", "gemini"}
    assert [f["time"] for f in session["feedback"]] == \
        sorted(f["time"] for f in session["feedback"]), "in playing order"
    # and on the bars the digest shows, not the ones each writer stored
    assert [f["bar"] for f in session["feedback"]] == \
        [feedback.bar_of(session, f["time"]) for f in session["feedback"]]
    assert [f["bar"] for f in session["feedback"]][:2] == [19, 27]

    from fourfloor import session as session_mod
    assert session_mod.validate(session) == [], "the field must stay additive"


def test_syncing_a_session_that_was_never_written_is_a_no_op(tmp_path: Path) -> None:
    lib = store.Library(tmp_path)
    rid, d = lib.create_remix()
    feedback.add_marker(d, 3.0, "good", "", rid)      # no session file at all
    assert feedback.sync_session(d) is None
    assert feedback.read(d, rid)["markers"][0]["note"] == ""


def test_an_author_from_the_browser_is_reduced_to_a_slug(remix) -> None:
    _, rid, d = remix
    m = feedback.apply(d, {"marker": {
        "time": 1.0, "category": "good", "author": "  Some OTHER/Ear!! "}}, rid)
    assert m["markers"][-1]["author"] == "some-other-ear"
    assert feedback.author_label("some-other-ear") == "Some Other Ear"
    assert feedback.author_label("gemini") == "Gemini"
    assert feedback.author_label("") == "Neel"


# ---------------------------------------------------------------------------
# the reference-pair lookup
# ---------------------------------------------------------------------------

def test_a_reference_remix_is_found_by_slug_not_by_path(tmp_path: Path,
                                                        monkeypatch) -> None:
    pairs = tmp_path / "refs" / "pairs"
    pairs.mkdir(parents=True)
    real = pairs / "Lofi 7.remix.mp3"
    real.write_bytes(b"x")
    monkeypatch.setenv("FOURFLOOR_REFS", str(tmp_path / "refs"))
    lib = store.Library(tmp_path / "home")

    for spelling in ("Lofi 7", "lofi-7", "LOFI_7", "lofi7"):
        assert lib.reference_pair(spelling)[0] == real.resolve(), spelling
    assert lib.reference_pair("something else") is None
    assert lib.reference_pair("") is None


@pytest.mark.parametrize("name", ["../../../etc/passwd", "..", "/etc/passwd",
                                  "Lofi 7/../../x"])
def test_a_reference_name_can_never_escape_the_pairs_folder(tmp_path: Path,
                                                            monkeypatch, name) -> None:
    pairs = tmp_path / "refs" / "pairs"
    pairs.mkdir(parents=True)
    (pairs / "Lofi 7.remix.mp3").write_bytes(b"x")
    (tmp_path / "secret.remix.mp3").write_bytes(b"secret")
    monkeypatch.setenv("FOURFLOOR_REFS", str(tmp_path / "refs"))
    lib = store.Library(tmp_path / "home")
    found = lib.reference_pair(name)
    assert found is None or pairs.resolve() in found[0].parents


def test_a_reference_must_be_audio_we_play(tmp_path: Path, monkeypatch) -> None:
    pairs = tmp_path / "refs" / "pairs"
    pairs.mkdir(parents=True)
    (pairs / "Lofi 7.remix.txt").write_text("not audio")
    monkeypatch.setenv("FOURFLOOR_REFS", str(tmp_path / "refs"))
    assert store.Library(tmp_path / "home").reference_pair("Lofi 7") is None


def test_no_pairs_folder_is_simply_no_reference(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FOURFLOOR_REFS", str(tmp_path / "nothing-here"))
    assert store.Library(tmp_path / "home").reference_pair("Lofi 7") is None
